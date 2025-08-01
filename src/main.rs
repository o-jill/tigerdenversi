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

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const HIDDENSIZE : i64 = weight::N_HIDDEN as i64;
const HIDDENSIZE2 : i64 = weight::N_HIDDEN2 as i64;
const MIN_COSANEAL : f64 = 1e-4;

fn net(vs : &nn::Path) -> impl Module {
    let relu = true;
    // let relu = false;  // sigmoid
    if relu {
        println!("activation function: RELU");
        nn::seq()
            .add(
                nn::linear(vs / "layer1",
                INPUTSIZE, HIDDENSIZE, Default::default()))
                // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
            .add_fn(|xs| xs.relu())
            .add(
                nn::linear(vs / "layer2",
                 HIDDENSIZE, HIDDENSIZE2, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer3", HIDDENSIZE2, 1, Default::default()))
    } else {
        println!("activation function: Sigmoid");
        nn::seq()
            .add(
                nn::linear(vs / "layer1",
                INPUTSIZE, HIDDENSIZE, Default::default()))
                // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
            .add_fn(|xs| xs.sigmoid())
            .add(
                nn::linear(vs / "layer2",
                 HIDDENSIZE, HIDDENSIZE2, Default::default()))
            .add_fn(|xs| xs.sigmoid())
            .add(nn::linear(vs / "layer3", HIDDENSIZE2, 1, Default::default()))
    }
}

fn loadtensor(vs : &mut VarStore, key : &str, src : &Tensor) {
    let mut val = vs.variables_.lock();
    let s = val.as_mut().unwrap().named_variables.get_mut(key).unwrap();
    s.set_data(src);
}

// load from `fname` into `vs`.
//
// .safetensor and ruversi weight format are available.
fn load(vs : &mut VarStore, weights_org : &weight::Weight, progress : usize)
        -> Result<(), String> {
    const INPSIZE : usize = bitboard::CELL_2D + 1 + 2;
    const HIDSIZE : usize =  HIDDENSIZE as usize;
    let wban = weights_org.wban(progress);
    let wtbn = weights_org.wteban(progress);
    let wfs = weights_org.wfixedstones(progress);
    let wdc = weights_org.wibias(progress);
    let whdn = weights_org.wlayer1(progress);
    let wdc2 = weights_org.wl1bias(progress);
    let whdn2 = weights_org.wlayer2(progress);
    let wdc3 = weights_org.wl2bias(progress);

    // layer1.weight
    let mut weights = [0.0f32 ; INPSIZE * HIDSIZE];
    for i in 0..HIDDENSIZE as usize {
        weights[i * INPSIZE..i * INPSIZE + bitboard::CELL_2D].copy_from_slice(
            &wban[i * bitboard::CELL_2D .. (i + 1) * bitboard::CELL_2D]);
        weights[i * INPSIZE + bitboard::CELL_2D] = wtbn[i];
        weights[i * INPSIZE + bitboard::CELL_2D + 1] = wfs[i];
        weights[i * INPSIZE + bitboard::CELL_2D + 2] = wfs[i + HIDSIZE];
    }
    let wl1 = Tensor::from_slice(&weights).view((HIDDENSIZE, INPUTSIZE));
    loadtensor(vs, "layer1.weight", &wl1);

    // layer1.bias
    let mut bias = [0.0f32 ; HIDDENSIZE as usize];
    bias.copy_from_slice(wdc);
    let wb1 = Tensor::from_slice(&bias).view(HIDDENSIZE);
    loadtensor(vs, "layer1.bias", &wb1);

    // layer2.weight
    let mut weights = [0.0f32 ; (HIDDENSIZE2 * HIDDENSIZE) as usize];
    weights.copy_from_slice(whdn);
    let wl2 = Tensor::from_slice(&weights).view((HIDDENSIZE2, HIDDENSIZE));
    loadtensor(vs, "layer2.weight", &wl2);

    // layer2.bias
    let mut bias = [0.0f32 ; HIDDENSIZE2 as usize];
    bias.copy_from_slice(wdc2);
    let wb1 = Tensor::from_slice(&bias).view(HIDDENSIZE2);
    loadtensor(vs, "layer2.bias", &wb1);

    // layer3.weight
    let mut weights = [0.0f32 ; HIDDENSIZE2 as usize];
    weights.copy_from_slice(whdn2);
    let wl2 = Tensor::from_slice(&weights).view((1, HIDDENSIZE2));
    loadtensor(vs, "layer3.weight", &wl2);

    // layer3.bias
    let bias = [wdc3 ; 1];
    let wb2 = Tensor::from_slice(&bias).view(1);
    loadtensor(vs, "layer3.bias", &wb2);

    Ok(())
}

fn storeweights(weights_dst : &mut weight::Weight, vs : VarStore, progress : usize) {
    println!("save to weights[{progress}]");

    // VarStore to weights
    let weights = vs.variables();
    let mut outp = [0.0f32 ; weight::N_WEIGHT];
    let mut tmp = [0.0f32 ; (INPUTSIZE * HIDDENSIZE) as usize];
    // let mut params = weight::EvalFile::V8.to_str().to_string() + "\n";

    let l1w = weights.get("layer1.weight").unwrap();
    println!("layer1.weight:{:?}", l1w.size());
    let numel = l1w.numel();
    l1w.copy_data(tmp.as_mut_slice(), numel);
    for i in 0..HIDDENSIZE as usize {
        let offset_out = i * bitboard::CELL_2D;
        let offset = i * INPUTSIZE as usize;
        outp[offset_out..offset_out + bitboard::CELL_2D].copy_from_slice(
            &tmp[offset..offset + bitboard::CELL_2D]);
        outp[weight::N_WEIGHT_TEBAN + i] = tmp[bitboard::CELL_2D + offset];
        outp[weight::N_WEIGHT_FIXST_B + i] =
            tmp[bitboard::CELL_2D + 1 + offset];
        outp[weight::N_WEIGHT_FIXST_W + i] =
            tmp[bitboard::CELL_2D + 2 + offset];
    }
    let keys = [
        "layer1.bias", "layer2.weight", "layer2.bias", "layer3.weight"
    ];
    let mut offset = weight::N_WEIGHT_INPUTBIAS;
    for key in keys {
        let l1w = weights.get(key).unwrap();
        println!("{key}:{:?}", l1w.size());
        let numel = l1w.numel();
        l1w.copy_data(tmp.as_mut_slice(), numel);
        outp[offset..offset + numel].copy_from_slice(&tmp[0..numel]);
        offset += numel;
    }
    let l3b = weights.get("layer3.bias").unwrap();
    println!("layer3.bias:{:?}", l3b.size());
    let numel = l3b.numel();
    l1w.copy_data(tmp.as_mut_slice(), numel);
    *outp.last_mut().unwrap() = tmp[0];

    println!("save to weight [{progress}]");
    weights_dst.copy_from_slice(&outp, progress);
}

fn writeweights(weights : &weight::Weight) {
    println!("save to weights.txt");
    if let Err(err) = weights.writev9("weights.txt") {
        panic!("{err}");
    }
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

fn main() -> Result<(), tch::TchError> {
    let arg = argument::Arg::parse();

    // let kifupath = "./kifu";
    // let mut boards = loadkifu(&findfiles(kifupath));
    let trainingpart = arg.partlist();
    let kifudir = arg.kifudir.unwrap_or("kifu".to_string()).clone();
    let devtype = arg.device.unwrap_or("cpu".to_string());
    let devtype = devtype.clone();
    let mut weights = weight::Weight::default();
    if let Some(awei) = arg.weight {
        println!("load weight from {}", &awei);
        if let Err(err) = weights.read(&awei) {
            panic!("{err}");
        }
    }

    let warmup = arg.warmup;
    for (prgs, en) in trainingpart.iter().enumerate() {
        if !*en {
            println!("progress[{prgs}] skipped.");
            continue;
        }

        println!("part[{prgs}]");
        let mut boards = kifudir.split(",").flat_map(
            |d| data_loader::loadkifu(&data_loader::findfiles(&format!("./{d}")), d, prgs)
            ).collect();

        data_loader::dedupboards(&mut boards);
        boards.shuffle(&mut rand::thread_rng());
        let testratio = arg.testratio as i64;

        let input = tch::Tensor::from_slice(
            &data_loader::extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
        println!("input : {} {:?}", input.dim(), input.size());

        let target = tch::Tensor::from_slice(
            &data_loader::extractscore(&boards)).view((boards.len() as i64, 1));
        println!("target: {} {:?}", target.dim(), target.size());

        let inputs = input.chunk(testratio, 0);
        let targets = target.chunk(testratio, 0);

        let device = if devtype == "mps" && tch::utils::has_mps() {
            Device::Mps      //  9h9m46s/100ep
            // 2m36s/100ep/18907b
        } else if devtype == "cuda" && tch::utils::has_cuda() {
            Device::Cuda(0)
        } else {
            Device::Cpu      // 13m0s/100ep/1516288b
            // 13s/100ep/18907b
        };
        let mut vs = VarStore::new(device);
        let nnet = net(&vs.root());

        if let Err(err) = load(&mut vs, &weights, prgs) {
            panic!("{err}");
        }

        let eta = arg.eta;
        let mut optm = nn::AdamW::default().build(&vs, eta)?;
        optm.set_weight_decay(arg.wdecay);
        for (key, t) in vs.variables().iter_mut() {
            println!("{key}:{:?}", t.size());
        }
        let period = arg.anealing;
        let autostop = arg.autostop;
        let datasize = target.size()[0];
        println!("datasize: {datasize}");
        let minibatch = if (datasize as i64) < 100 * testratio * arg.minibatch {
                ((datasize  as i64 / 100 / testratio + 15) / 16) * 16
            } else {
                arg.minibatch
            };
        let minibatch = if minibatch > 0 {
                minibatch
            } else {
                4
            };
        println!("epoch:{}", arg.epoch);
        println!("eta:{eta}");
        println!("cosine aneaing:{period}");
        println!("mini batch: {}", minibatch);
        println!("weight decay:{}", arg.wdecay);
        println!("test ratio:{testratio}");
        println!("auto stop:{autostop:?}");
        println!("training part: {trainingpart:?}");
        println!("warmup: {warmup}");
        let start = std::time::Instant::now();
        if warmup > 1 {
            for wep in 0..warmup {
                let w_eta_min = eta * MIN_COSANEAL;
                let a = (eta - w_eta_min) / warmup as f64;
                optm.set_lr(w_eta_min + a * wep as f64);

                let iloss = if inputs.len() > 1 {wep % inputs.len()} else {99999};
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
                let elapsed = start.elapsed();
                print!("{}", &epochspeed(wep, arg.epoch + warmup, testloss, elapsed));
                std::io::stdout().flush().unwrap();
            }
        }
        if period > 1 {
            let mut sum_loss_prev = 99999999.9;
            let mut sum_loss = 0.0;
            for ep in 0..arg.epoch {
                let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
                optm.set_lr(
                    eta * MIN_COSANEAL +
                        eta * 0.5 * (1.0 - MIN_COSANEAL)
                            * (1.0 + (std::f64::consts::PI * (ep as i32 % period) as f64 / (period - 1) as f64).cos())
                );
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
                let elapsed = start.elapsed();
                print!("{}", &epochspeed(ep + warmup, arg.epoch + warmup, testloss, elapsed));
                std::io::stdout().flush().unwrap();
                if let Some(threshold) = autostop {
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
        } else {
            for ep in 0..arg.epoch {
                let iloss = if inputs.len() > 1 {ep % inputs.len()} else {99999};
                for ((i, inp), tar) in inputs.iter().enumerate().zip(targets.iter()) {
                    if i == iloss {continue;}

                    let mut dataset = Iter2::new(inp, tar, arg.minibatch);
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
                let elapsed = start.elapsed();
                print!("{}", &epochspeed(ep, arg.epoch, testloss, elapsed));
                std::io::stdout().flush().unwrap();
            }
        }
        println!();

        // VarStore to weights
        storeweights(&mut weights, vs, prgs);
    }

    writeweights(&weights);

    Ok(())
}
