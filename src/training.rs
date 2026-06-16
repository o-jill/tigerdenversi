use super::*;
use chrono::Utc;
use std::time::Duration;
use tch::nn::{self, OptimizerConfig, VarStore};
use tch::{Device, data::Iter2, Kind, Tensor};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const MIN_COSANEAL : f64 = 1e-4;
enum LargeRatioIndex {
    Div = 0,
    Train = 1,
    Eval = 2,
    Size = 3,
}

#[derive(Clone, Debug, PartialEq)]
enum Part {
    Off = 0,
    On = 1,
    Large = 2,
}

impl std::convert::From<&str> for Part {
    fn from(txt: &str) -> Self {
        if txt.is_empty() {return Part::Off;}

        let disable = [/*"", */"0", "false", "no", "none", "off", "zero"];
        let large = "large";

        let txt_lo = txt.to_lowercase();

        if disable.contains(&txt_lo.as_str()) {
            Part::Off
        } else if txt_lo == large {
            Part::Large
        } else {
            Part::On
        }
    }
}

impl Part {
    pub fn is_off(&self) -> bool {
        *self == Part::Off
    }

    #[allow(dead_code)]
    pub fn is_on(&self) -> bool {
        *self == Part::On
    }

    pub fn is_large(&self) -> bool {
        *self == Part::Large
    }
}

pub struct Training {
    trainingpart : Vec<Part>,
    kifudir : Vec<String>,
    matedir : Vec<String>,
    matefiles : Vec<String>,
    devtype : String,
    device : tch::Device,
    autostop : Option<f64>,
    epoch : usize,
    eta : f64,
    minibatch : i64,
    period : i32,
    anealing_step : i32,
    stopwatch : std::time::Instant,
    testratio : i64,
    warmup : usize,
    wdecay : f64,
    awdecay : f64,
    weights : weight::Weight,
    multibar : MultiProgress,
    log : std::fs::File,
    loss_curve : Vec<f64>,
    show_progressbar : bool,
    show_graph : bool,
    large_dir : String,
    large_ratio : [i32 ; LargeRatioIndex::Size as usize],
}

impl std::fmt::Display for Training {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "")
    }
}

impl From<argument::Arg> for Training {
    fn from(arg : argument::Arg) -> Self {
        let strdt = Utc::now().format("%Y%m%d%H%M%S").to_string();
        let path = if let Some(path) = arg.log {
                let path = path.replace("<DATETIME>", &strdt);
                let invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
                if path.chars().any(|c| invalid_chars.contains(&c)) {
                    panic!("path:{path} contains invalid letter!");
                }
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
        let kifudir = if arg.kifudir.is_empty() {
                vec![String::from("kifu") ; 1]
            } else {
                arg.kifudir
            };
        let matedir = arg.matedir;
        let matefiles = arg.mate_file;
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
                format!("load weight from {awei}").as_bytes()).unwrap();
            if let Err(err) = weights.read(&awei) {
                panic!("{err}");
            }
        }
        let large_dir = arg.large_dir.unwrap_or_default();
        let mut large_ratio = [0 ; 3];
        if arg.large_ratio.len() == LargeRatioIndex::Size as usize {
            large_ratio.copy_from_slice(&arg.large_ratio);
        }

        Self {
            trainingpart : partlist,
            kifudir,
            matedir,
            matefiles,
            devtype,
            device,
            autostop : arg.autostop,
            epoch : arg.epoch,
            eta : arg.eta,
            minibatch : arg.minibatch,
            period : arg.anealing,
            anealing_step : 0,
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
            show_progressbar : arg.progressbar,
            show_graph : arg.graph,
            large_dir,
            large_ratio
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
    fn partlist(part : &Option<String>) -> Vec<Part> {
        let mut ret = vec![Part::On ; weight::N_PROGRESS_DIV];
        if part.is_none() {return ret;}

        let txt = part.as_ref().unwrap();
        if txt.is_empty() {return ret;}

        // println!("txt_lo:{txt_lo}, txt:[{txt}]");
        for (i, elem) in txt.split(',').enumerate() {
            // println!("elem:[{elem}]");
            if i >= weight::N_PROGRESS_DIV {break;}

            ret[i] = elem.into();
        }
        ret
    }

    fn anealing_learning_rate(&mut self, ep : usize) -> (f64, bool) {
        let mut caperiod = self.anealing_step;  // # of finished cycles
        let mut period = self.period * (1 << caperiod);  // period for current step
        let mut offset = (0..caperiod).fold(0usize, |a, x| a + self.period  as usize * (1 << x));
        let mut eta = self.eta * (1.0 - self.awdecay).powi(caperiod);
        let next_step = ep - offset == period as usize;
        // self.putlog(&format!("ep:{ep}, cycles:{caperiod}, period:{period}, offset:{offset}"));
        if next_step {
            // move on to next period
            self.anealing_step += 1;
            caperiod = self.anealing_step;  // # of finished cycles
            period = self.period * (1 << caperiod);  // period for current step
            offset = (0..caperiod).fold(0usize, |a, x| a + self.period as usize * (1 << x));
            eta = self.eta * (1.0 - self.awdecay).powi(caperiod);

            self.putlog(&format!(
                "next_step: ep:{ep}, cycles:{caperiod}, period:{period}, offset:{offset}, eta:{eta}"));
        }

        (eta * MIN_COSANEAL +
            eta * 0.5 * (1.0 - MIN_COSANEAL)
                * (1.0 + (std::f64::consts::PI * (ep - offset) as f64
                    / (period - 1) as f64).cos()),
            next_step)
    }

    fn epochspeed(
        ep : usize, maxepoch : usize, loss : f64, elapsed : std::time::Duration) -> String {
        let epoch = ep + 1;
        let elapsedsec = elapsed.as_secs_f64();
        let speed = elapsedsec / epoch as f64;

        let etasecs = (maxepoch as f64 - epoch as f64) * speed;

        format!("ep:{epoch:4}/{maxepoch} loss:{loss:.3} ")
        + & if etasecs > 0.0 {
            let esthour = (etasecs / 3600.0) as i32;
            let estmin = ((etasecs - esthour as f64 * 3600.0) / 60.0) as i32;
            let estsec = (etasecs % 60.0) as i32;
            format!("ETA:{esthour:02}h{estmin:02}m{estsec:02}s ")
        } else {
            "ETA:--h--m--s ".to_string()
        }
        + & if speed > 3600.0 * 1.1 {
            format!("{:.1}hour/epoch", speed / 3600.0)
        } else  if speed > 99.0 {
            format!("{:.1}min/epoch", speed / 60.0)
        } else {
            format!("{speed:.1}sec/epoch")
        }
        + &format!(" {elapsedsec:.0}sec\n")
    }

    fn prepare_data(&mut self, progress : usize, pb : &Option<ProgressBar>)
            -> (tch::Tensor, tch::Tensor) {
        // let sta = std::time::Instant::now();
        let mut boards : Vec<_> = self.kifudir.iter().flat_map(
            |d| {
                if let Some(pb) = pb {
                    pb.inc(1);
                    let path = std::path::Path::new(d);
                    let fname = path.components().rev().find_map(|c|
                        match c {
                        std::path::Component::Normal(os_str) => {Some(os_str)},
                        _ => {None},
                        }).unwrap_or_default();
                    pb.set_message(format!("dir:{}", fname.to_string_lossy()));
                }
                data_loader::loadkifu(
                    &data_loader::find_kifu_files(d),
                    d, progress, &mut self.log, pb.is_none())}
            ).collect();

        if !self.matedir.is_empty() {
            for d in self.matedir.iter() {
                if let Some(pb) = pb {
                    pb.inc(1);
                    let path = std::path::Path::new(d);
                    let fname = path.components().rev().find_map(|c|
                        match c {
                        std::path::Component::Normal(os_str) => {Some(os_str)},
                        _ => {None},
                        }).unwrap_or_default();
                    pb.set_message(format!("dir:{}", fname.to_string_lossy()));
                }
                let mut brds = data_loader::find_mate_files(d).iter().flat_map(
                    |fname| {
                    // onli "mate*" are available.
                    if !fname.starts_with("mate") {return Vec::new();}

                    let path = std::path::Path::new(d).join(fname);
                    match data_loader::load_mates(&path.display().to_string(), progress) {
                        Ok(arr) => {arr},
                        Err(msg) => {panic!("{msg}")},
                    }
                }).collect::<Vec<_>>();
                if !brds.is_empty() {boards.append(&mut brds);}
            }
        }

        if !self.matefiles.is_empty() {
            let mut mates = self.matefiles.iter().flat_map(
                |path| {
                    if let Some(pb) = pb {pb.inc(1);}
                    match data_loader::load_mates(path, progress) {
                        Ok(arr) => {arr},
                        Err(msg) => {panic!("{msg}")},
                    }
                }
            ).collect::<Vec<_>>();
            self.putlog(&format!("mates : {:?} size:{}", self.matefiles, mates.len()));
            if !mates.is_empty() {boards.append(&mut mates);}
        }

        data_loader::dedupboards(&mut boards, &mut self.log, pb.is_none());
        boards.shuffle(&mut rand::thread_rng());
        // println!("{}msec",sta.elapsed().as_millis());
        if let Some(pb) = pb {pb.inc(1);}

        // eliminate boards because of RAM size
        const LARGE_NUMBER_OF_BOARDS : usize = 1024 * 1024 * 128;
        if boards.len() > LARGE_NUMBER_OF_BOARDS {
            self.putlog(&format!(
                "board size:{} exceeds limit({LARGE_NUMBER_OF_BOARDS})",
                boards.len()));
            boards.truncate(LARGE_NUMBER_OF_BOARDS);
            boards.shrink_to_fit();
        }

        let input = tch::Tensor::from_slice(
            &data_loader::extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
        self.putlog(&format!("input : {} {:?}", input.dim(), input.size()));

        let target = tch::Tensor::from_slice(
            &data_loader::extractscore(&boards)).view((boards.len() as i64, 1));
        self.putlog(&format!("target: {} {:?}", target.dim(), target.size()));
        if let Some(pb) = pb {pb.inc(1);}

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

    fn prepare_large_dataset(&mut self, indexes : &[usize], progress : usize, pb : &Option<ProgressBar>)
            -> (tch::Tensor, tch::Tensor) {
        // let sta = std::time::Instant::now();
        let mut boards : Vec<_> = indexes.iter().flat_map(|&index| {
            if let Some(pb) = pb {
                pb.inc(1);
                pb.set_message(format!("loading idx {index}"));
            }

            let dirpath = std::path::Path::new(&self.large_dir);
            let path = data_loader::gen_div_file_name(dirpath, index);
            data_loader::load_mates(path.to_str().unwrap(), progress)
                .unwrap_or_else(|e| {
                    panic!("{e} in load mates in prepare_large_dataset")
                })
        }).collect::<Vec<(bitboard::BitBoard, i8)>>();

        data_loader::dedupboards(&mut boards, &mut self.log, pb.is_none());
        // boards.shuffle(&mut rand::thread_rng());
        // println!("{}msec",sta.elapsed().as_millis());
        if let Some(pb) = pb {pb.inc(1);}

        // eliminate boards because of RAM size
        // const LARGE_NUMBER_OF_BOARDS : usize = 1024 * 1024 * 128;
        // if boards.len() > LARGE_NUMBER_OF_BOARDS {
        //     self.putlog(&format!(
        //         "board size:{} exceeds limit({LARGE_NUMBER_OF_BOARDS})",
        //         boards.len()));
        //     boards.truncate(LARGE_NUMBER_OF_BOARDS);
        //     boards.shrink_to_fit();
        // }

        let input = tch::Tensor::from_slice(
            &data_loader::extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
        self.putlog(&format!("input : {} {:?}", input.dim(), input.size()));

        let target = tch::Tensor::from_slice(
            &data_loader::extractscore(&boards)).view((boards.len() as i64, 1));
        self.putlog(&format!("target: {} {:?}", target.dim(), target.size()));

        if let Some(pb) = pb {pb.inc(1);}

        (input, target)
    }

    fn warmup_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        if self.is_not_warmup() {return;}

        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(self.warmup as u64));
            pb.set_style(
                ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🔥🔥🪵"));
            Some(pb)
        } else {
            None
        };
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
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
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
            self.update(testloss, &pb, wep, elapsed);
        }
        if let Some(pb) = pb {
            pb.finish_with_message(format!("warm up - done! final loss:{final_loss:.3}"));
        }
    }

    fn warmup_sequence_large(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            minibatch : i64, progress : usize) {
        if self.is_not_warmup() {return;}

        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(
                self.warmup as u64 * (1 + 3 * (self.large_training_size() + self.large_eval_size())) as u64));
            pb.set_style(
                ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🔥🔥🪵"));
            Some(pb)
        } else {
            None
        };

        // let testratio = inputs.len();
        let mut final_loss = 0f64;

        for wep in 0..self.warmup {
            let (train_idx, eval_idx) = self.gen_largedata_index().unwrap();

            let eta = self.eta;
            let w_eta_min = eta * MIN_COSANEAL;
            let a = (eta - w_eta_min) / self.warmup as f64;
            optm.set_lr(w_eta_min + a * wep as f64);

            for i in 0..train_idx.len() {
                let (inputs, targets) =
                    self.prepare_large_dataset(
                        &train_idx[i..i + 1], progress, &pb);

                if let Some(pb) = &pb {
                    pb.set_message(format!("{i}/{} ep:{wep}/{}", train_idx.len(), self.warmup));
                }
                let mut dataset = Iter2::new(&inputs, &targets, minibatch);
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
                for (xs, ys) in dataset {
                    // println!("xs: {} {:?} ys: {} {:?}",
                    //          xs.dim(), xs.size(), ys.dim(), ys.size());
                    let loss =
                        nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                    optm.backward_step(&loss);
                }
            }

            let testloss = if eval_idx.is_empty() {
                    0f64
                } else {
                    let (inputs, targets) =
                        self.prepare_large_dataset(&eval_idx, progress, &pb);
                    let loss = nnet.forward(&inputs)
                            .mse_loss(&targets, tch::Reduction::Mean);
                    loss.double_value(&[])
                };

            let elapsed = self.elapsed();

            final_loss = testloss;
            self.loss_curve.push(testloss);
            self.update(testloss, &pb, wep, elapsed);
        }

        if let Some(pb) = &pb {
            pb.finish_with_message(format!("warm up - done! final loss:{final_loss:.3}"));
        }
    }

    fn update(&mut self, loss : f64,
        pb : &Option<ProgressBar>, ep : usize, elapsed : Duration) {
        if let Some(pb) = pb {
            pb.set_message(format!("loss: {loss:.3}"));
            pb.inc(1);
        }

        self.putlog(&Self::epochspeed(
            ep, self.epoch + self.warmup, loss, elapsed));
    }

    fn cos_anealing_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(self.epoch as u64));
            pb.set_style(
                ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("📗📖📓"));
            Some(pb)
        } else {
            None
        };
        let mut sum_loss_prev = 99999999.9;
        let mut sum_loss = 0.0;
        let mut final_loss = 0f64;
        let testratio = inputs.len();
        let mut actual_epochs = self.epoch;
        self.anealing_step = 0;
        for ep in 0..self.epoch * 2 {
            let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
            let (new_lr, next_step) = self.anealing_learning_rate(ep);
            // stop automatically after desinated epoch
            // and after learning w/ minimum learning rate.
            if next_step && ep >= self.epoch {
                actual_epochs = ep;
                // self.putlog(&format!(
                //     "next_step && ep >= self.epoch: {next_step} {ep}"));
                break;
            }

            optm.set_lr(new_lr);
            for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                if i == iloss {continue;}

                let mut dataset = Iter2::new(inp, tar, minibatch);
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
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
            self.update(testloss, &pb, ep + self.warmup, elapsed);
            // 学習を始めたけどロスが大きいときはなにかおかしい
            if ep >  5 {
                // 少なくとも300を超えてるときはおかしい
                const WARNING_THRESHOLD : f64 = 300f64;
                if testloss > WARNING_THRESHOLD {
                    let msg = format!("loss:{testloss}, ep:{ep}, lr:{new_lr}");
                    eprintln!("{msg}");
                    self.putlog(&msg);
                    let msg = format!(
                        "ep: {ep} iloss: {iloss} input_mean: {:.3} target_mean: {:.3}",
                        inputs[iloss].mean(Kind::Float).double_value(&[]),
                        targets[iloss].mean(Kind::Float).double_value(&[]));
                    eprintln!("{msg}");
                    self.putlog(&msg);
                    panic!("{msg}");
                }
            }

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
        if let Some(pb) = pb {
            pb.set_length(actual_epochs as u64);
            pb.set_position(actual_epochs as u64);
            pb.finish_with_message(
                format!("cos anealing - done! final loss:{final_loss:.3}"));
        }
    }

    fn cos_anealing_sequence_large(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            minibatch : i64, progress : usize) {
        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(
                self.epoch as u64 * (1 + 3 * (self.large_training_size() + self.large_eval_size())) as u64));
            pb.set_style(
                ProgressStyle::with_template(
                "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("📗📖📓"));
            Some(pb)
        } else {
            None
        };
        let mut sum_loss_prev = 99999999.9;
        let mut sum_loss = 0.0;
        let mut final_loss = 0f64;
        let testratio = self.testratio as usize;
        let mut actual_epochs = self.epoch;
        self.anealing_step = 0;
        for ep in 0..self.epoch * 2 {
            let (train_idx, eval_idx) = self.gen_largedata_index().unwrap();

            let iloss = if testratio > 1 {ep % testratio} else {99999};
            let (new_lr, next_step) = self.anealing_learning_rate(ep);
            // stop automatically after desinated epoch
            // and after learning w/ minimum learning rate.
            if next_step && ep >= self.epoch {
                actual_epochs = ep;
                // self.putlog(&format!(
                //     "next_step && ep >= self.epoch: {next_step} {ep}"));
                break;
            }

            optm.set_lr(new_lr);
            for i in 0..train_idx.len() {
                let (inputs, targets) =
                    self.prepare_large_dataset(
                        &train_idx[i..i + 1], progress, &pb);

                if let Some(pb) = &pb {
                    pb.set_message(format!("{i}/{} ep:{ep}/{}", train_idx.len(), self.epoch));
                }
                let mut dataset = Iter2::new(&inputs, &targets, minibatch);
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
                for (xs, ys) in dataset {
                    // println!("xs: {} {:?} ys: {} {:?}",
                    //          xs.dim(), xs.size(), ys.dim(), ys.size());
                    let loss =
                        nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                    optm.backward_step(&loss);
                }
            }
            let testloss = if eval_idx.is_empty() {
                    0f64
                } else {
                    let (inputs, targets) =
                        self.prepare_large_dataset(&eval_idx, progress, &pb);
                    let loss = nnet.forward(&inputs)
                            .mse_loss(&targets, tch::Reduction::Mean);
                    loss.double_value(&[])
                };
            let elapsed = self.elapsed();
            final_loss = testloss;
            self.loss_curve.push(testloss);
            self.update(testloss, &pb, ep + self.warmup, elapsed);
            // 学習を始めたけどロスが大きいときはなにかおかしい
            if ep >  5 {
                // 少なくとも300を超えてるときはおかしい
                const WARNING_THRESHOLD : f64 = 300f64;
                if testloss > WARNING_THRESHOLD {
                    let msg = format!("loss:{testloss}, ep:{ep}, lr:{new_lr}");
                    eprintln!("{msg}");
                    self.putlog(&msg);
                    let msg = format!(
                        "ep: {ep} iloss: {iloss} input_mean: ??? target_mean: ???");
                    // let msg = format!(
                    //     "ep: {ep} iloss: {iloss} input_mean: {:.3} target_mean: {:.3}",
                    //     inputs[iloss].mean(Kind::Float).double_value(&[]),
                    //     targets[iloss].mean(Kind::Float).double_value(&[]));
                    eprintln!("{msg}");
                    self.putlog(&msg);
                    // panic!("{msg}");
                }
            }

            if self.autostop.is_none() {continue;}

            let threshold = self.autostop.unwrap();
            sum_loss += testloss;
            let testratio = self.testratio;
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
        if let Some(pb) = pb {
            pb.set_length(actual_epochs as u64);
            pb.set_position(actual_epochs as u64);
            pb.finish_with_message(
                format!("cos anealing - done! final loss:{final_loss:.3}"));
        }
    }

    fn std_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(self.epoch as u64));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {wide_bar} [{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("📗📖📓"));
            Some(pb)
        } else {
            None
        };
        let testratio = inputs.len();
        for ep in 0..self.epoch {
            let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
            for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                if i == iloss {continue;}

                let mut dataset = Iter2::new(inp, tar, minibatch);
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
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
            self.update(testloss, &pb, ep, elapsed);
        }
    }

    fn std_sequence_large(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            minibatch : i64, progress : usize) {
        let pb = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(self.epoch as u64));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {wide_bar} [{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("📗📖📓"));
            Some(pb)
        } else {
            None
        };
        let testratio = self.testratio as usize;
        for ep in 0..self.epoch {
            let (train_idx, eval_idx) = self.gen_largedata_index().unwrap();

            for i in 0..train_idx.len() {
                let (inputs, targets) =
                    self.prepare_large_dataset(
                        &train_idx[i..i + 1], progress, &pb);

                let mut dataset = Iter2::new(&inputs, &targets, minibatch);
                let dataset = if vs.device() == Device::Cpu {
                    dataset.shuffle()
                } else {
                    dataset.shuffle().to_device(vs.device())
                };
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
                let (inputs, targets) =
                    self.prepare_large_dataset(&eval_idx, progress, &pb);
                let loss = nnet.forward(&inputs)
                        .mse_loss(&targets, tch::Reduction::Mean);
                loss.double_value(&[])
            };
            let elapsed = self.elapsed();
            self.loss_curve.push(testloss);
            self.update(testloss, &pb, ep, elapsed);
        }
    }

    fn start_time(&mut self) {
        self.stopwatch = std::time::Instant::now();
    }

    fn elapsed(&self) -> std::time::Duration {
        self.stopwatch.elapsed()
    }

    /// 学習データを複数のファイルに分割してから学習を始めるかどうか
    fn is_large_data_mode(&self) -> bool {
        let div = self.large_ratio[LargeRatioIndex::Div as usize];
        let train = self.large_ratio[LargeRatioIndex::Train as usize];
        let eval = self.large_ratio[LargeRatioIndex::Eval as usize];
        !self.large_dir.is_empty()  // 出力先の指定あり
            && div > 1  // 2つ以上に分ける
            && train > 0  // 1つは絶対に必要
            && eval > 0  // 1つは絶対に必要
            && div >= train + eval
    }

    fn large_division_size(&self) -> usize {
        self.large_ratio[LargeRatioIndex::Div as usize] as usize
    }

    fn large_training_size(&self) -> usize {
        self.large_ratio[LargeRatioIndex::Train as usize] as usize
    }

    fn large_eval_size(&self) -> usize {
        self.large_ratio[LargeRatioIndex::Eval as usize] as usize
    }

    /// generate indexes for training and loss evaluation.
    ///
    /// # Returns
    /// (indexes for training, indexes for loss evaluation)
    fn gen_largedata_index(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        let div_ratio = self.large_division_size();
        let train_file_size = self.large_training_size();
        let eval_file_size = self.large_eval_size();

        if div_ratio == 0 || div_ratio < train_file_size + eval_file_size {
            return None;
        }

        let mut numbers = (0..div_ratio).collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        numbers.shuffle(&mut rng);

        let train_idx = numbers[0..train_file_size].to_vec();
        let eval_idx = numbers[train_file_size..train_file_size + eval_file_size].to_vec();

        Some((train_idx, eval_idx))
    }

    /// run training
    pub fn run(&mut self) -> Result<(), tch::TchError> {
        let pbtop = if self.show_progressbar {
            let pb = self.multibar.add(
            ProgressBar::new(weight::N_PROGRESS_DIV as u64));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len}").unwrap());

            Some(pb)
        } else {
            None
        };

        let partlist = self.trainingpart.clone();
        for (progress, p) in partlist.iter().enumerate() {
            if let Some(pb ) = &pbtop {pb.inc(1);}

            if p.is_off() {
                let msg = format!("progress[{progress}] skipped.");
                println!("{msg}");
                self.putlog(&msg);
                continue;
            }

            if p.is_large() && self.is_large_data_mode() {
                self.run_large_dataset(progress, &pbtop)?;
            } else {
                self.run_normal(progress, &pbtop)?;
            }
        }
        if let Some(pb ) = &pbtop {pb.finish();}

        neuralnet::writeweights(&self.weights);

        self.plot_loss();

        Ok(())
    }

    /// clean-up destination directory, load kifu and mate files
    /// and separate them into files.
    ///
    /// - destination directory: `self.large_dir`
    /// - source kifu files: `self.kifudir`
    /// - mate files: `self.matedir` and`self.matefiles`
    /// - \# of division: `self.large_ratio[LargeRatioIndex::Div]`
    fn split_large_dataset(&mut self, pbsplit : &Option<ProgressBar>) -> Result<(), String> {
        // clean up
        self.putlog("clean_up_large_data");
        if let Some(pb) = pbsplit {
            pb.inc(1);
            pb.set_message("cleaning up large data ...");
        }
        data_loader::clean_up_large_data(&self.large_dir)?;

        let div_ratio = self.large_division_size() as i32;

        // load kifu
        self.putlog("load_large_kifu");
        if let Some(pb) = pbsplit {
            pb.inc(1);
            pb.set_message("loading large kifu ...");
        }
        for kifudir in self.kifudir.iter() {
            data_loader::prepare_large_data(
                kifudir, &self.large_dir, div_ratio)?;
        }

        // load mate
        self.putlog("load_large_mate");
        if let Some(pb) = pbsplit {
            pb.inc(1);
            pb.set_message("loading large mate ...");
        }
        data_loader::prepare_large_mate(
            &self.matedir, &self.matefiles, &self.large_dir, div_ratio)?;

        Ok(())
    }

    /// 学習データを複数のファイルに分割してから学習するモード
    ///
    /// # Arguments
    /// - `progress`: progress of the game.
    /// - `pbtop`: progressbar
    fn run_large_dataset(
        &mut self, progress : usize, pbtop : &Option<ProgressBar>)
            -> Result<(), tch::TchError> {

        self.anealing_step = 0;

        let pbchild = if pbtop.is_some() {
            let pb = self.multibar.add(
                ProgressBar::new(
                    {
                        let steps =
                            self.matedir.len() + self.kifudir.len() + 7;
                        steps + self.matefiles.len()
                    } as u64));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🪵🪓🌴"));
            pb.set_message("loading data...");
            Some(pb)
        } else {
            None
        };

        self.putlog("large mode:");
        // prepare dataset
        self.split_large_dataset(&pbchild).map_err(tch::TchError::Torch)?;

        if let Some(pb) = &pbchild {pb.inc(1);}

        let mut vs = VarStore::new(self.device);
        let nnet = neuralnet::net(&vs.root());

        if let Err(err) = neuralnet::load(&mut vs, &self.weights, progress) {
            panic!("{err}");
        }

        if let Some(pb) = &pbchild {pb.inc(1);}
        let mut optm = nn::AdamW::default().build(&vs, self.eta)?;
        optm.set_weight_decay(self.wdecay);

        self.putlog(&
            vs.variables().iter().map(|(key, t)| {
                format!("{key}:{:?}\n", t.size())
            }).collect::<Vec<String>>().join(""));
        let datasize = 0;  // target.size()[0];

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
        if let Some(pb) = &pbchild {
            pb.finish_with_message(
                format!("preparing {progress} - done!"));
        }

        self.start_time();

        if self.is_cos_anealing() {  // cos anealing
            self.warmup_sequence_large(
                &nnet, &mut vs, &mut optm, minibatch, progress);

            self.cos_anealing_sequence_large(
                &nnet, &mut vs, &mut optm, minibatch, progress);
        } else {
            self.std_sequence_large(
                &nnet, &mut vs, &mut optm, minibatch, progress);
        }

        // VarStore to weights
        neuralnet::storeweights(&mut self.weights, vs, progress);

        Ok(())
    }

    /// 一括でデータを読み込んでおいてそれを学習するモード
    ///
    /// # Arguments
    /// - `progress`: progress of the game.
    /// - `pbtop`: progressbar
    fn run_normal(&mut self, progress : usize, pbtop : &Option<ProgressBar>)
             -> Result<(), tch::TchError> {
        self.anealing_step = 0;

        let pbchild = if pbtop.is_some() {
            let pb = self.multibar.add(
                ProgressBar::new(
                    {
                        let steps =
                            self.matedir.len() + self.kifudir.len() + 7;
                        steps + self.matefiles.len()
                    } as u64));
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}]{wide_bar}[{eta_precise}] {pos}/{len} {msg}").unwrap()
                .progress_chars("🪵🪓🌴"));
            pb.set_message("loading data...");
            Some(pb)
        } else {
            None
        };
        let (input, target) = self.prepare_data(progress, &pbchild);
        let inputs = input.chunk(self.testratio, 0);
        let targets = target.chunk(self.testratio, 0);
        if let Some(pb) = &pbchild {pb.inc(1);}

        let mut vs = VarStore::new(self.device);
        let nnet = neuralnet::net(&vs.root());

        if let Err(err) = neuralnet::load(&mut vs, &self.weights, progress) {
            panic!("{err}");
        }

        if let Some(pb) = &pbchild {pb.inc(1);}
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
        if let Some(pb) = &pbchild {
            pb.finish_with_message(
                format!("preparing {progress} - done!"));
        }

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

        // VarStore to weights
        neuralnet::storeweights(&mut self.weights, vs, progress);

        Ok(())
    }

    pub fn write(&self) {
        neuralnet::writeweights(&self.weights);
    }

    fn putlog(&mut self, msg : &str) {
        let msg = if msg.ends_with("\n") {
            msg
        } else {
            &(msg.to_string() + "\n")
        };
        self.log.write_all(msg.as_bytes()).unwrap();
        self.log.sync_all().unwrap();
        if !self.show_progressbar {
            print!("{msg}");
            std::io::stdout().flush().unwrap();
        }
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
    assert_eq!(
        p1, vec![Part::On, Part::On, Part::On, Part::On, Part::On, Part::On]);

    let s2 = Some(String::from("1,,0"));
    let p2 = Training::partlist(&s2);
    assert_eq!(p2,
        vec![Part::On, Part::Off, Part::Off, Part::On, Part::On, Part::On]);

    let s3 = Some(String::from("-1,false,zero,no,none,off"));
    let p3 = Training::partlist(&s3);
    assert_eq!(p3,
        vec![Part::On, Part::Off, Part::Off, Part::Off, Part::Off, Part::Off]);

    // let s4 = Some(String::from("no,none,off"));
    // let p4 = Training::partlist(&s4);
    // assert_eq!(p4, vec![Part::Off, Part::Off, Part::Off]);

    let s5 = Some(String::from("no,none,a,0"));
    let p5 = Training::partlist(&s5);
    assert_eq!(p5,
        vec![Part::Off, Part::Off, Part::On, Part::Off, Part::On, Part::On]);

    let s6 = Some(String::from("0,"));
    let p6 = Training::partlist(&s6);
    assert_eq!(p6,
        vec![Part::Off, Part::Off, Part::On, Part::On, Part::On, Part::On]);

    let s7 = Some(String::from(",,,,large,large"));
    let p7 = Training::partlist(&s7);
    assert_eq!(p7,
        vec![Part::Off, Part::Off, Part::Off, Part::Off, Part::Large, Part::Large]);
}

#[test]
fn test_gen_largedata_index() {
    let mut arg = argument::Arg::parse();
    arg.large_dir = Some("aaa".to_string());
    arg.large_ratio = vec![10, 5, 2];
    let t = Training::from(arg);
    assert!(t.is_large_data_mode());
    let (a, b) = t.gen_largedata_index().unwrap();
    assert_eq!(a.len(), 5);
    assert_eq!(b.len(), 2);

    let mut arg = argument::Arg::parse();
    arg.large_dir = Some("aaa".to_string());
    arg.large_ratio = vec![10, 5, 5];
    let t = Training::from(arg);
    assert!(t.is_large_data_mode());
    let (a, b) = t.gen_largedata_index().unwrap();
    assert_eq!(a.len(), 5);
    assert_eq!(b.len(), 5);

    let c = [a.as_slice(), b.as_slice()].concat();

    for i in 0..10 {
        assert!(c.contains(&i));
    }
}

#[test]
fn test_gen_largedata_index_invalid() {
    let arg = argument::Arg::parse();
    let t = Training::from(arg);
    assert!(!t.is_large_data_mode());
    assert_eq!(t.gen_largedata_index(), None);

    let mut arg = argument::Arg::parse();
    arg.large_ratio = vec![10, 5, 2];
    let t = Training::from(arg);
    assert!(!t.is_large_data_mode());
    let (a, b) = t.gen_largedata_index().unwrap();
    assert_eq!(a.len(), 5);
    assert_eq!(b.len(), 2);

    let mut arg = argument::Arg::parse();
    arg.large_dir = Some("aaa".to_string());
    let t = Training::from(arg);
    assert!(!t.is_large_data_mode());
    assert_eq!(t.gen_largedata_index(), None);

    let mut arg = argument::Arg::parse();
    arg.large_dir = Some("aaa".to_string());
    arg.large_ratio = vec![10, 5, 6];
    let t = Training::from(arg);
    assert!(!t.is_large_data_mode());
    assert_eq!(t.gen_largedata_index(), None);

    let mut arg = argument::Arg::parse();
    arg.large_dir = Some("aaa".to_string());
    arg.large_ratio = vec![10, 7, 4];
    let t = Training::from(arg);
    assert!(!t.is_large_data_mode());
    assert_eq!(t.gen_largedata_index(), None);
}
