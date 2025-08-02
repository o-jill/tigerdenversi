use super::*;

use std::io::Write;
use tch::nn::{self, OptimizerConfig, VarStore};
use tch::{Device, data::Iter2, Tensor};

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
    weights : weight::Weight,
}

impl std::fmt::Display for Training {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "")
    }
}

impl From<argument::Arg> for Training {
    fn from(arg : argument::Arg) -> Self {
        let partlist = arg.partlist();
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
            println!("load weight from {}", &awei);
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
            weights,
        }
    }
}

impl Training {
    fn anealing_learning_rate(&self, ep : usize) -> f64 {
        let eta = self.eta;
        let period = self.period;
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

    fn prepare_data(&self, progress : usize) -> (tch::Tensor, tch::Tensor) {
        let mut boards = self.kifudir.split(",").flat_map(
            |d| data_loader::loadkifu(
                &data_loader::findfiles(&format!("./{d}")), d, progress)
            ).collect();

        data_loader::dedupboards(&mut boards);
        boards.shuffle(&mut rand::thread_rng());

        let input = tch::Tensor::from_slice(
            &data_loader::extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
        println!("input : {} {:?}", input.dim(), input.size());

        let target = tch::Tensor::from_slice(
            &data_loader::extractscore(&boards)).view((boards.len() as i64, 1));
        println!("target: {} {:?}", target.dim(), target.size());

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
        if self.warmup < 1 {return;}

        let testratio = inputs.len();
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
            print!("{}", &Self::epochspeed(
                    wep, self.epoch + self.warmup, testloss, elapsed));
            std::io::stdout().flush().unwrap();
        }
    }

    fn cos_anealing_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
        let mut sum_loss_prev = 99999999.9;
        let mut sum_loss = 0.0;
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
            print!("{}", &Self::epochspeed(
                    ep + self.warmup, self.epoch + self.warmup,
                    testloss, elapsed));
            std::io::stdout().flush().unwrap();

            if self.autostop.is_none() {continue;}

            let threshold = self.autostop.unwrap();
            sum_loss += testloss;
            if (ep + 1) % (testratio as i32 * self.period) as usize == 0 {
                println!("\nsum_loss{}:{sum_loss}", ep + 1);

                if  sum_loss_prev - sum_loss > threshold {
                    sum_loss_prev = sum_loss;
                    sum_loss = 0.0;
                } else {
                    println!("done as a result of learning enough.");
                    break;
                }
            }
        }
    }

    fn std_sequence(&mut self, nnet : &impl nn::Module,
            vs : &mut VarStore, optm : &mut tch::nn::Optimizer,
            inputs : &[Tensor], targets : &[Tensor], minibatch : i64) {
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
            print!("{}", &Self::epochspeed(ep, self.epoch, testloss, elapsed));
            std::io::stdout().flush().unwrap();
        }
    }

    fn start_time(&mut self) {
        self.stopwatch = std::time::Instant::now();
    }

    fn elapsed(&self) -> std::time::Duration {
        self.stopwatch.elapsed()
    }

    pub fn run(&mut self) -> Result<(), tch::TchError> {
        let trainingpart = self.trainingpart.clone();
        for (progress, en) in trainingpart.iter().enumerate() {
            if !*en {
                println!("progress[{progress}] skipped.");
                continue;
            }

            println!("part[{progress}]");

            let (input, target) = self.prepare_data(progress);
            let inputs = input.chunk(self.testratio, 0);
            let targets = target.chunk(self.testratio, 0);

            let mut vs = VarStore::new(self.device);
            let nnet = neuralnet::net(&vs.root());

            if let Err(err) = neuralnet::load(&mut vs, &self.weights, progress) {
                panic!("{err}");
            }

            let mut optm = nn::AdamW::default().build(&vs, self.eta)?;
            optm.set_weight_decay(self.wdecay);

            for (key, t) in vs.variables().iter_mut() {
                println!("{key}:{:?}", t.size());
            }
            let datasize = target.size()[0];
            
            let minibatch = self.adjust_minibatch(datasize);
            
            println!("auto stop:{:?}", self.autostop);
            println!("datasize: {datasize}");
            println!("devtype: {}", self.devtype);
            println!("cosine aneaing:{}", self.period);
            println!("epoch:{}", self.epoch);
            println!("eta:{}", self.eta);
            println!("mini batch: {minibatch}");
            println!("test ratio:{}", self.testratio);
            println!("training part: {:?}", self.trainingpart);
            println!("warmup: {}", self.warmup);
            println!("weight decay:{}", self.wdecay);

            self.start_time();

            if self.period > 1 {  // cos anealing
                self.warmup_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);

                self.cos_anealing_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);
            } else {
                self.std_sequence(
                    &nnet, &mut vs, &mut optm, &inputs, &targets, minibatch);
            }
            println!();

            // VarStore to weights
            neuralnet::storeweights(&mut self.weights, vs, progress);
        }

        neuralnet::writeweights(&self.weights);

        Ok(())
    }

    pub fn write(&self) {
        neuralnet::writeweights(&self.weights);
    }
}
