use super::*;

use std::io::Write;
use std::time::Duration;
use tch::nn::{self, OptimizerConfig, VarStore};
use tch::{Device, data::Iter2, Tensor};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const MIN_COSANEAL : f64 = 1e-4;

pub struct Training {
    trainingpart : Vec<bool>,
    kifudir : String,
    devtype : String,
    device : tch::Device,
    autostop : Option<f64>,
    epoch : usize,
    eta : f64,
    minibatch : i64,
    period : i32,
    stopwatch : std::time::Instant,
    testratio : i64,
    warmup : usize,
    wdecay : f64,
    awdecay : f64,
    weights : weight::Weight,
    multibar : MultiProgress,
    log : std::fs::File,
    loss_curve : Vec<f64>,
    show_graph : bool,
}

impl std::fmt::Display for Training {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "")
    }
}

impl From<argument::Arg> for Training {
    fn from(arg : argument::Arg) -> Self {
        let path = if let Some(path) = arg.log {
            path
        } else {
            if cfg!(target_os="windows") {
                String::from("nul")
            } else {
                String::from("/dev/null")
            }
        };
        let mut log = match std::fs::File::create(path) {
        Ok(f) => {f},
        Err(e) => {panic!("{e}")},
        };

        let partlist = Self::partlist(&arg.part);
        let kifudir = arg.kifudir.unwrap_or("kifu".to_string()).clone();
        let devtype = arg.device.unwrap_or("cpu".to_string());
        let devtype = devtype.clone();
        let device    = if devtype == "mps" && tch::utils::has_mps() {
                Device::Mps
            } else if devtype == "cuda" && tch::utils::has_cuda() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            };

        let mut weights = weight::Weight::default();
        if let Some(awei) = arg.weight {
            log.write_all(
                format!("load weight from {}", &awei).as_bytes()).unwrap();
            if let Err(err) = weights.read(&awei) {
                panic!("{err}");
            }
        }

        Self {
            trainingpart : partlist,
            kifudir,
            devtype,
            device,
            autostop : arg.autostop,
            epoch : arg.epoch,
            eta : arg.eta,
            minibatch : arg.minibatch,
            period : arg.anealing,
            stopwatch : std::time::Instant::now(),
            testratio : arg.testratio as i64,
            warmup : arg.warmup,
            wdecay : arg.wdecay,
            awdecay : arg.awdecay,
            weights,
            multibar : MultiProgress::new(),
            log,
            loss_curve :
                Vec::with_capacity(
                    weight::N_PROGRESS_DIV * (arg.warmup + arg.epoch)),
            show_graph : arg.graph,
        }
    }
}

impl Training {
    /// returns if cos-anealing mode or not.
    fn is_cos_anealing(&self) -> bool {
        self.period > 1
    }

    /// returns if warm-up mode(false) or not(true).
    fn is_not_warmup(&self) -> bool {
        self.warmup < 1
    }

    /// csv text to get an array if each part will be trained or not.
    /// "", "0", "false", "no", "none", "off" and "zero" disables training.
    ///
    /// ex. "" becomes [true, true, true]
    /// ex. "1,,0" becomes [true, false, false]
    /// ex. "-1,false,zero" becomes [true, false, false]
    fn partlist(part : &Option<String>) -> Vec<bool> {
        let mut ret = vec![true ; weight::N_PROGRESS_DIV];
        if part.is_none() {return ret;}

        let txt = part.as_ref().unwrap();
        if txt.is_empty() {return ret;}

        let disable = [/*"", */"0", "false", "no", "none", "off", "zero"];
        let txt_lo = txt.to_lowercase();
        println!("txt_lo:{txt_lo}, txt:[{txt}]");
        for (i, elem) in txt_lo.split(',').enumerate() {
            println!("elem:[{elem}]");
            if i >= weight::N_PROGRESS_DIV {break;}
            if elem.is_empty() {
                ret[i] = false;
                continue;
            }

            ret[i] = !disable.contains(&elem);
        }
        ret
    }

    fn anealing_learning_rate(&self, ep : usize) -> f64 {
        let period = self.period;
        let caperiod = ep as i32 / period;
        let eta = self.eta * (1.0 - self.awdecay).powi(caperiod);
        eta * MIN_COSANEAL +
            eta * 0.5 * (1.0 - MIN_COSANEAL)
                * (1.0 + (std::f64::consts::PI * (ep as i32 % period) as f64
                    / (period - 1) as f64).cos())
    }

    fn epochspeed(
        ep : usize, maxepoch : usize, loss : f64, elapsed : std::time::Duration) -> String {
        let epoch = ep + 1;
        let speed = elapsed.as_secs_f64() / (epoch) as f64;

        let etasecs = (maxepoch - epoch) as f64 * speed;

        let esthour = (etasecs / 3600.0) as i32;
        let estmin = ((etasecs - esthour as f64 * 3600.0) / 60.0) as i32;
        let estsec = (etasecs % 60.0) as i32;

        let mut res = format!("ep:{epoch:4}/{maxepoch} loss:{loss:.3} ");
        res += &format!("ETA:{esthour:02}h{estmin:02}m{estsec:02}s ");
        res + &if speed > 3600.0 * 1.1 {
                format!("{:.1}hour/epoch\n", speed / 3600.0)
            } else  if speed > 99.0 {
                format!("{:.1}min/epoch\n", speed / 60.0)
            } else {
                format!("{speed:.1}sec/epoch\n")
            }
    }

    fn prepare_data(&mut self, progress : usize, pb : &ProgressBar)
            -> (tch::Tensor, tch::Tensor) {
        // let sta = std::time::Instant::now();
        let mut boards = self.kifudir.split(",").flat_map(
            |d| {
                pb.inc(1);
                data_loader::loadkifu(
                    &data_loader::findfiles(&format!("./{d}")),
                    d, progress, &mut self.log)}
            ).collect();

        data_loader::dedupboards(&mut boards, &mut self.log);
        boards.shuffle(&mut rand::thread_rng());
        // println!("{}msec",sta.elapsed().as_millis());
        pb.inc(1);

        let input = tch::Tensor::from_slice(
            &data_loader::extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
        self.putlog(&format!("input : {} {:?}\n", input.dim(), input.size()));

        let target = tch::Tensor::from_slice(
            &data_loader::extractscore(&boards)).view((boards.len() as i64, 1));
        self.putlog(&format!("target: {} {:?}\n", target.dim(), target.size()));
        pb.inc(1);

        (input, target)
    }

    fn adjust_minibatch(&self, datasize : i64) -> i64 {
        let minibatch = self.minibatch;
        let testratio = self.testratio;
        let minibatch = if datasize < 100 * testratio * minibatch {
                    ((datasize / 100 / testratio + 15) / 16) * 16
                } else {
                    minibatch
                };
        if minibatch > 0 {
            minibatch
        } else {
            4
        }
    }

    fn warmup_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        if self.is_not_warmup() {return;}

        let pb = self.multibar.add(
            ProgressBar::new(self.warmup as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🔥🔥🪵"));
        let testratio = inputs.len();
        let mut final_loss = 0f64;
        for wep in 0..self.warmup {
            let w_eta_min = self.eta * MIN_COSANEAL;
            let a = (self.eta - w_eta_min) / self.warmup as f64;
            optm.set_lr(w_eta_min + a * wep as f64);

            let iloss = if inputs.len() > 1 {wep % testratio} else {99999};
            for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                if i == iloss {continue;}

                let mut dataset = Iter2::new(inp, tar, minibatch);
                // let dataset = dataset.shuffle();
                let dataset = dataset.shuffle().to_device(vs.device());
                for (xs, ys) in dataset {
                    // println!("xs: {} {:?} ys: {} {:?}",
                    //          xs.dim(), xs.size(), ys.dim(), ys.size());
                    let loss =
                        nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                    optm.backward_step(&loss);
                }
            }
            let testloss = if testratio == 0 {
                    0f64
                } else {
                    let loss = nnet.forward(&inputs[iloss])
                            .mse_loss(&targets[iloss], tch::Reduction::Mean);
                    loss.double_value(&[])
                };
            let elapsed = self.elapsed();
            final_loss = testloss;
            self.loss_curve.push(testloss);
            self.update(testloss, Some(&pb), wep, elapsed);
        }
        pb.finish_with_message(format!("warm up - done! final loss:{final_loss:.3}"));
    }

    fn update(&mut self, loss : f64,
        pb : Option<&ProgressBar>, ep : usize, elapsed : Duration) {
        if let Some(pb) = pb {
            pb.set_message(format!("loss: {loss:.3}"));
            pb.inc(1);
        }

        self.log.write_all(Self::epochspeed(
            ep, self.epoch + self.warmup, loss, elapsed).as_bytes()).unwrap();
    }

    fn cos_anealing_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        let pb = self.multibar.add(
            ProgressBar::new(self.epoch as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
            .progress_chars("📗📖📓"));
        let mut sum_loss_prev = 99999999.9;
        let mut sum_loss = 0.0;
        let mut final_loss = 0f64;
        let testratio = inputs.len();
        for ep in 0..self.epoch {
            let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
            optm.set_lr(self.anealing_learning_rate(ep));
            for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                if i == iloss {continue;}

                let mut dataset = Iter2::new(inp, tar, minibatch);
                // let dataset = dataset.shuffle();
                let dataset = dataset.shuffle().to_device(vs.device());
                for (xs, ys) in dataset {
                    // println!("xs: {} {:?} ys: {} {:?}",
                    //          xs.dim(), xs.size(), ys.dim(), ys.size());
                    let loss =
                        nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                    optm.backward_step(&loss);
                }
            }
            let testloss = if testratio == 0 {
                    0f64
                } else {
                    let loss = nnet.forward(&inputs[iloss])
                            .mse_loss(&targets[iloss], tch::Reduction::Mean);
                    loss.double_value(&[])
                };
            let elapsed = self.elapsed();
            final_loss = testloss;
            self.loss_curve.push(testloss);
            self.update(testloss, Some(&pb), ep + self.warmup, elapsed);

            if self.autostop.is_none() {continue;}

            let threshold = self.autostop.unwrap();
            sum_loss += testloss;
            if (ep + 1) % (testratio as i32 * self.period) as usize == 0 {
                let msg = format!("sum_loss{}:{sum_loss}", ep + 1);
                self.putlog(&msg);

                if  sum_loss_prev - sum_loss > threshold {
                    sum_loss_prev = sum_loss;
                    sum_loss = 0.0;
                } else {
                    println!("done as a result of learning enough.");
                    break;
                }
            }
        }
        pb.finish_with_message(format!("cos anealing - done! final loss:{final_loss:.3}"));
    }

    fn std_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        let pb = self.multibar.add(
            ProgressBar::new(self.epoch as u64));
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {wide_bar} [{eta_precise}] {pos}/{len} {msg}").unwrap()
            .progress_chars("📗📖📓"));
        let testratio = inputs.len();
        for ep in 0..self.epoch {
            let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
            for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                if i == iloss {continue;}

                let mut dataset = Iter2::new(inp, tar, minibatch);
                // let dataset = dataset.shuffle();
                let dataset = dataset.shuffle().to_device(vs.device());
                // let mut loss = tch::Tensor::new();
                for (xs, ys) in dataset {
                    // println!("xs: {} {:?} ys: {} {:?}",
                    //          xs.dim(), xs.size(), ys.dim(), ys.size());
                    let loss =
                        nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                    optm.backward_step(&loss);
                }
            }
            let testloss = if testratio == 0 {
                0f64
            } else {
                let loss = nnet.forward(&inputs[iloss])
                        .mse_loss(&targets[iloss], tch::Reduction::Mean);
                loss.double_value(&[])
            };
            let elapsed = self.elapsed();
            self.loss_curve.push(testloss);
            self.update(testloss, Some(&pb), ep, elapsed);
        }
    }

    fn start_time(&mut self) {
        self.stopwatch = std::time::Instant::now();
    }

    fn elapsed(&self) -> std::time::Duration {
        self.stopwatch.elapsed()
    }

    pub fn run(&mut self) -> Result<(), tch::TchError> {
        let pb = self.multibar.add(
            ProgressBar::new(weight::N_PROGRESS_DIV as u64));
        let trainingpart = self.trainingpart.clone();
        for (progress, en) in trainingpart.iter().enumerate() {
            pb.inc(1);
            if !*en {
                let msg = format!("progress[{progress}] skipped.");
                println!("{msg}");
                self.putlog(&msg);
                continue;
            }
            let pb = self.multibar.add(
                ProgressBar::new(
                    self.kifudir.chars().fold(6,
                        |acc, c| if c == ',' {acc + 1} else {acc})));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🪵🪓🌴"));
            pb.set_message("loading data...");
            let (input, target) = self.prepare_data(progress, &pb);
            let inputs = input.chunk(self.testratio, 0);
            let targets = target.chunk(self.testratio, 0);
            pb.inc(1);

            let mut vs = VarStore::new(self.device);
            let nnet = neuralnet::net(&vs.root());

            if let Err(err) = neuralnet::load(&mut vs, &self.weights, progress) {
                panic!("{err}");
            }

            pb.inc(1);
            let mut optm = nn::AdamW::default().build(&vs, self.eta)?;
            optm.set_weight_decay(self.wdecay);

            self.putlog(&
                vs.variables().iter().map(|(key, t)| {
                    format!("{key}:{:?}\n", t.size())
                }).collect::<Vec<String>>().join(""));
            let datasize = target.size()[0];
            
            let minibatch = self.adjust_minibatch(datasize);
            
            let msg = format!("auto stop:{:?}\n", self.autostop)
                + &format!("datasize: {datasize}\n")
                + &format!("devtype: {}\n", self.devtype)
                + &format!("cosine aneaing:{}\n", self.period)
                + &format!("epoch:{}\n", self.epoch)
                + &format!("eta:{}\n", self.eta)
                + &format!("mini batch: {minibatch}\n")
                + &format!("test ratio:{}\n", self.testratio)
                + &format!("training part: {:?}\n", self.trainingpart)
                + &format!("warmup: {}\n", self.warmup)
                + &format!("weight decay:{}\n", self.wdecay);
            self.putlog(&msg);
            pb.finish_with_message(
                format!("preparing {progress} - done!"));

            self.start_time();

            if self.is_cos_anealing() {  // cos anealing
                self.warmup_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);

                self.cos_anealing_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);
            } else {
                self.std_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);
            }
            // println!();

            // VarStore to weights
            neuralnet::storeweights(&mut self.weights, vs, progress);
        }
        pb.finish();

        neuralnet::writeweights(&self.weights);

        self.plot_loss();

        Ok(())
    }

    pub fn write(&self) {
        neuralnet::writeweights(&self.weights);
    }

    fn putlog(&mut self, msg : &str) {
        self.log.write_all(msg.as_bytes()).unwrap();
    }

    fn plot_loss(&self) {
        if !self.show_graph {return;}
        if self.loss_curve.is_empty() {
            panic!("self.loss_curve.is_empty()");
        }

        let w = if self.is_cos_anealing() {
            self.warmup + self.epoch
        } else {
            self.epoch
        };
        let data =
                (0..weight::N_PROGRESS_DIV).map(|i|
                    self.loss_curve[i * w..(i + 1) * w]
                        .to_vec()).collect::<Vec<Vec<f64>>>();
        // println!("{} {} {} {}",
        //.    data.len(), data[0].len(), data[1].len(), data[2].len());
        println!("{}",
            rasciigraph::plot_many(
                data,
                rasciigraph::Config::default()
                    // .with_offset(100)
                    .with_height(10)
                    .with_width(40)
                    .with_caption("loss history".to_string())
                ));
    }
}

#[test]
fn test_partlist() {
    // csv text to get an array if each part will be trained or not.
    // "", "0", "false", "no", "none", "off" and "zero" disables training.
    //
    // ex. "" becomes [true, true, true]
    // ex. "1,,0" becomes [true, false, false]
    // ex. "-1,false,zero" becomes [true, false, false]
    // fn partlist(part : &Option<String>) -> Vec<bool>

    let s1 = Some(String::new());
    let p1 = Training::partlist(&s1);
    assert_eq!(p1, vec![true, true, true]);

    let s2 = Some(String::from("1,,0"));
    let p2 = Training::partlist(&s2);
    assert_eq!(p2, vec![true, false, false]);

    let s3 = Some(String::from("-1,false,zero"));
    let p3 = Training::partlist(&s3);
    assert_eq!(p3, vec![true, false, false]);

    let s4 = Some(String::from("no,none,off"));
    let p4 = Training::partlist(&s4);
    assert_eq!(p4, vec![false, false, false]);

    let s5 = Some(String::from("no,none,a,0"));
    let p5 = Training::partlist(&s5);
    assert_eq!(p5, vec![false, false, true]);

    let s6 = Some(String::from("0,"));
    let p6 = Training::partlist(&s6);
    assert_eq!(p6, vec![false, false, true]);
}
