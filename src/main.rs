use std::io::Write;
use clap::Parser;
use rand::seq::SliceRandom;
use tch::nn::{self, Module, OptimizerConfig, VarStore};
use tch::{Device, data::Iter2, Tensor};

mod kifu;
mod bitboard;
mod weight;
mod argument;
mod data_loader;
mod neuralnet;

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const MIN_COSANEAL : f64 = 1e-4;

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

fn prepare_data(kifudir : &str, progress : usize) -> (tch::Tensor, tch::Tensor) {
    let mut boards = kifudir.split(",").flat_map(
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

fn adjust_minibatch(minibatch : i64, datasize : i64, testratio : i64) -> i64 {
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

fn anealing_learning_rate(eta : f64, ep : usize, period : i32) -> f64 {
    eta * MIN_COSANEAL +
        eta * 0.5 * (1.0 - MIN_COSANEAL)
            * (1.0 + (std::f64::consts::PI * (ep as i32 % period) as f64 / (period - 1) as f64).cos())
}

fn warmup_sequence(nnet : &impl nn::Module,
        vs : &mut VarStore, optm : &mut tch::nn::Optimizer, inputs : &[Tensor], targets : &[Tensor],
        elapsedtimer : &std::time::Instant,
        warmup : usize, eta : f64, minibatch : i64, epochs : usize) {
    if warmup < 1 {return;}

    let testratio = inputs.len();
    for wep in 0..warmup {
        let w_eta_min = eta * MIN_COSANEAL;
        let a = (eta - w_eta_min) / warmup as f64;
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
        let elapsed = elapsedtimer.elapsed();
        print!("{}", &epochspeed(wep, epochs + warmup, testloss, elapsed));
        std::io::stdout().flush().unwrap();
    }
}

fn cos_anealing_sequence(nnet : &impl nn::Module,
        vs : &mut VarStore, optm : &mut tch::nn::Optimizer, inputs : &[Tensor], targets : &[Tensor],
        elapsedtimer : &std::time::Instant,
        warmup : usize, eta : f64, minibatch : i64, epochs : usize, period : i32, autostop : Option<f64>) {
    let mut sum_loss_prev = 99999999.9;
    let mut sum_loss = 0.0;
    let testratio = inputs.len();
    for ep in 0..epochs {
        let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
        optm.set_lr(anealing_learning_rate(eta, ep, period));
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
        let elapsed = elapsedtimer.elapsed();
        print!("{}", &epochspeed(ep + warmup, epochs + warmup, testloss, elapsed));
        std::io::stdout().flush().unwrap();

        if autostop.is_none() {continue;}

        let threshold = autostop.unwrap();
        sum_loss += testloss;
        if (ep + 1) % (testratio as i32 * period) as usize == 0 {
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

fn std_sequence(nnet : &impl nn::Module,
        vs : &mut VarStore, optm : &mut tch::nn::Optimizer, inputs : &[Tensor], targets : &[Tensor],
        elapsedtimer : &std::time::Instant, minibatch : i64, epochs : usize) {
    let testratio = inputs.len();
    for ep in 0..epochs {
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
        let elapsed = elapsedtimer.elapsed();
        print!("{}", &epochspeed(ep, epochs, testloss, elapsed));
        std::io::stdout().flush().unwrap();
    }
}

fn main() -> Result<(), tch::TchError> {
    let arg = argument::Arg::parse();

    // let kifupath = "./kifu";
    // let mut boards = loadkifu(&findfiles(kifupath));
    let trainingpart = arg.partlist();
    let kifudir = arg.kifudir.unwrap_or("kifu".to_string()).clone();
    let devtype = arg.device.unwrap_or("cpu".to_string());
    let devtype = devtype.clone();
    let device = if devtype == "mps" && tch::utils::has_mps() {
        Device::Mps      //  9h9m46s/100ep
        // 2m36s/100ep/18907b
    } else if devtype == "cuda" && tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu      // 13m0s/100ep/1516288b
        // 13s/100ep/18907b
    };
    let mut weights = weight::Weight::default();
    if let Some(awei) = arg.weight {
        println!("load weight from {}", &awei);
        if let Err(err) = weights.read(&awei) {
            panic!("{err}");
        }
    }

    let autostop = arg.autostop;
    let eta = arg.eta;
    let period = arg.anealing;
    let testratio = arg.testratio as i64;
    let warmup = arg.warmup;
    for (prgs, en) in trainingpart.iter().enumerate() {
        if !*en {
            println!("progress[{prgs}] skipped.");
            continue;
        }

        println!("part[{prgs}]");

        let (input, target) = prepare_data(&kifudir, prgs);
        let inputs = input.chunk(testratio, 0);
        let targets = target.chunk(testratio, 0);

        let mut vs = VarStore::new(device);
        let nnet = neuralnet::net(&vs.root());

        if let Err(err) = neuralnet::load(&mut vs, &weights, prgs) {
            panic!("{err}");
        }

        let mut optm = nn::AdamW::default().build(&vs, eta)?;
        optm.set_weight_decay(arg.wdecay);

        for (key, t) in vs.variables().iter_mut() {
            println!("{key}:{:?}", t.size());
        }
        let datasize = target.size()[0];
        println!("datasize: {datasize}");

        let minibatch = adjust_minibatch(arg.minibatch, datasize, testratio);

        println!("epoch:{}", arg.epoch);
        println!("eta:{eta}");
        println!("cosine aneaing:{period}");
        println!("mini batch: {minibatch}");
        println!("weight decay:{}", arg.wdecay);
        println!("test ratio:{testratio}");
        println!("auto stop:{autostop:?}");
        println!("training part: {trainingpart:?}");
        println!("warmup: {warmup}");

        let start = std::time::Instant::now();

        if period > 1 {  // cos anealing
            warmup_sequence(
                &nnet, &mut vs, &mut optm, &inputs, &targets, &start,
                warmup, eta, minibatch, arg.epoch);

            cos_anealing_sequence(
                &nnet, &mut vs, &mut optm, &inputs, &targets, &start,
                warmup, eta, minibatch, arg.epoch, period, autostop);
        } else {
            std_sequence(
                &nnet, &mut vs, &mut optm, &inputs, &targets, &start,
                minibatch, arg.epoch);
        }
        println!();

        // VarStore to weights
        neuralnet::storeweights(&mut weights, vs, prgs);
    }

    neuralnet::writeweights(&weights);

    Ok(())
}
