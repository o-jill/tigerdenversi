use std::io::Write;

use clap::Parser;
use rand::seq::SliceRandom;
use tch::nn::{self, Module, OptimizerConfig, VarStore};
use tch::{Device, data::Iter2, Tensor};

mod kifu;
mod bitboard;
mod weight;

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const HIDDENSIZE : i64 = weight::N_HIDDEN as i64;
const HIDDENSIZE2 : i64 = weight::N_HIDDEN2 as i64;
const MIN_COSANEAL : f64 = 1e-4;

#[derive(Debug, Parser)]
#[command(version, author, about)]
struct Arg {
    /// path for weight.safetensor to use
    #[arg(short, long)]
    weight : Option<String>,
    /// initial learning rate
    #[arg(short, long, default_value_t = 0.01)]
    eta : f64,
    /// # of epochs
    #[arg(long, default_value_t = 100)]
    epoch : usize,
    /// kifu directory
    #[arg(long)]
    kifudir : Option<String>,
    /// mini batch size
    #[arg(long, default_value_t = 16)]
    minibatch : i64,
    /// storing weight after some iterations as weight.EPOCH.txt.
    #[arg(short, long)]
    progress : Option<String>,
    /// cosine anealing period.
    #[arg(short, long, default_value_t = 0)]
    anealing : i32,
    /// device to process. cuda, mps or cpu. default:cpu.
    #[arg(long)]
    device : Option<String>,
    /// weight decay
    #[arg(long, default_value_t = 0.0002)]
    wdecay : f64,
    /// ratio of test data for calc loss
    #[arg(long, default_value_t = 5)]
    testratio : usize,
    /// check if trained enough. [prefered: 0]
    #[arg(long)]
    autostop : Option<f64>,
}

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

// list up kifu
fn findfiles(kifupath : &str) -> Vec<String> {
    let dir = std::fs::read_dir(kifupath).unwrap();
    let mut files = dir.filter_map(|entry| {
        entry.ok().and_then(|e|
            e.path().file_name().map(|n|
                n.to_str().unwrap().to_string()
            )
        )}).collect::<Vec<String>>().iter().filter(|&fnm| {
            fnm.contains("kifu")
            // fnm.contains(".txt")
        }).cloned().collect::<Vec<String>>();
    // println!("{:?}", files);

    files.sort();
    files
}

fn loadkifu(files : &[String], d : &str) -> Vec<(bitboard::BitBoard, i8, i8, i8, i8)> {
    let mut boards : Vec<(bitboard::BitBoard, i8, i8, i8, i8)> = Vec::new();
    for fname in files.iter() {
        let path = format!("{d}/{}", fname);
        print!("{path}\r");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines:Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        for t in kifu.list.iter() {
            let ban = bitboard::BitBoard::from(&t.rfen).unwrap();
            if ban.is_full() {continue;}

            let (fsb, fsw) = ban.fixedstones();

            let ban90 = ban.rotate90();
            let ban180 = ban.rotate180();
            let ban270 = ban180.rotate90();
            let banh = ban.flip_horz();
            let banv = ban.flip_vert();
            let score = kifu.score.unwrap();
            boards.push((ban, t.teban, fsb, fsw, score));
            boards.push((ban90, t.teban, fsb, fsw, score));
            boards.push((ban180, t.teban, fsb, fsw, score));
            boards.push((ban270, t.teban, fsb, fsw, score));
            boards.push((banh, t.teban, fsb, fsw, score));
            boards.push((banv, t.teban, fsb, fsw, score));
        }
    }
    println!();
    boards
}

fn dedupboards(boards : &mut Vec<(bitboard::BitBoard, i8, i8, i8, i8)>) {
    boards.sort_by(|a, b| {
        a.0.black.cmp(&b.0.black).then(a.0.white.cmp(&b.0.white))
    });
    boards.dedup_by(|a, b| {a == b});
    println!("board: {} boards", boards.len());
}

fn extractboards(boards : &[(bitboard::BitBoard, i8, i8, i8, i8)])
        -> Vec<f32> {
    boards.iter().map(|(b, t, fb, fw, _s)| {
        let mut v = [0.0f32 ; INPUTSIZE as usize];
        for x in 0..8 {
            for y in 0..8 {
                v[x * bitboard::NUMCELL + y] = b.at(x as u8, y as u8) as f32;
            }
        }
        v[bitboard::CELL_2D] = *t as f32;
        v[bitboard::CELL_2D + 1] = *fb as f32;
        v[bitboard::CELL_2D + 2] = *fw as f32;
        v
    }).collect::<Vec<[f32 ; INPUTSIZE as usize]>>().concat()
}

fn extractscore(boards : &[(bitboard::BitBoard, i8, i8, i8, i8)]) -> Vec<f32> {
    boards.iter().map(|(_b, _t, _fb, _fw, s)| *s as f32).collect::<Vec<f32>>()
}

fn loadtensor(vs : &mut VarStore, key : &str, src : &Tensor) {
    let mut val = vs.variables_.lock();
    let s = val.as_mut().unwrap().named_variables.get_mut(key).unwrap();
    s.set_data(src);
}

// load from `fname` into `vs`.
//
// .safetensor and ruversi weight format are available.
fn load(fname : &str, vs : &mut VarStore) -> Result<(), String> {
    let path = std::path::Path::new(fname);
    if !path.exists() {
        return Err(format!("{fname} was not found..."));
    }
    if fname.ends_with(".safetensors") {
        println!("load weight: {}", fname);
        if let Err(e) = vs.load(fname) {
            return Err(e.to_string());
        }

        return Ok(());
    }

    let mut txtweight = weight::Weight::new();
    txtweight.read(fname)?;

    const INPSIZE : usize = bitboard::CELL_2D + 1 + 2;
    const HIDSIZE : usize =  HIDDENSIZE as usize;
    let wban = txtweight.wban();
    let wtbn = txtweight.wteban();
    let wfs = txtweight.wfixedstones();
    let wdc = txtweight.wibias();
    let whdn = txtweight.wlayer1();
    let wdc2 = txtweight.wl1bias();
    let whdn2 = txtweight.wlayer2();
    let wdc3 = txtweight.wl2bias();

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

fn storeweights(vs : VarStore) {
    println!("save to weight.safetensors");
    vs.save("weight.safetensors").unwrap();

    // VarStore to weights
    let weights = vs.variables();
    let mut outp = [0.0f32 ; (INPUTSIZE * HIDDENSIZE) as usize];
    // let mut params = weight::EvalFile::V8.to_str().to_string() + "\n";
    let mut params = weight::EvalFile::latest_header() + "\n";
    let mut paramste = String::new();
    let mut paramsfb = String::new();
    let mut paramsfw = String::new();

    let l1w = weights.get("layer1.weight").unwrap();
    println!("layer1.weight:{:?}", l1w.size());
    let numel = l1w.numel();
    l1w.copy_data(outp.as_mut_slice(), numel);
    for i in 0..HIDDENSIZE {
        let offset = (i * INPUTSIZE) as usize;
        params += &outp[offset..(offset + bitboard::CELL_2D)].iter()
            .map(|a| format!("{a},")).collect::<Vec<String>>().join("");
        paramste += &format!("{},", outp[bitboard::CELL_2D + offset]);
        paramsfb += &format!("{},", outp[bitboard::CELL_2D + 1 + offset]);
        paramsfw += &format!("{},", outp[bitboard::CELL_2D + 2 + offset]);
    }
    params += &paramste;
    params += &paramsfb;
    params += &paramsfw;
    // let keys = ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"];
    let keys = [
        "layer1.bias", "layer2.weight", "layer2.bias", "layer3.weight"
    ];
    for key in keys {
        let l1w = weights.get(key).unwrap();
        println!("{key}:{:?}", l1w.size());
        let numel = l1w.numel();
        l1w.copy_data(outp.as_mut_slice(), numel);
        params += &outp[0..numel].iter()
            .map(|a| format!("{a}")).collect::<Vec<String>>().join(",");
        params += ",";
    }
    let l1w = weights.get("layer3.bias").unwrap();
    println!("layer3.bias:{:?}", l1w.size());
    let numel = l1w.numel();
    l1w.copy_data(outp.as_mut_slice(), numel);
    params += &outp[0..numel].iter()
        .map(|a| format!("{a}")).collect::<Vec<String>>().join(",");
    println!("save to weight.txt");
    let mut f = std::fs::File::create("weights.txt").unwrap();
    f.write_all(params.as_bytes()).unwrap();
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
    let arg = Arg::parse();

    // let kifupath = "./kifu";
    // let mut boards = loadkifu(&findfiles(kifupath));
    let kifudir = arg.kifudir.unwrap_or(String::from("kifu"));
    let mut boards = kifudir.split(",").flat_map(
        |d| loadkifu(&findfiles(&format!("./{d}")), d)
        ).collect();

    dedupboards(&mut boards);
    boards.shuffle(&mut rand::thread_rng());
    let testratio = arg.testratio as i64;

    let input = tch::Tensor::from_slice(
        &extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
    println!("input : {} {:?}", input.dim(), input.size());

    let target = tch::Tensor::from_slice(
        &extractscore(&boards)).view((boards.len() as i64, 1));
    println!("target: {} {:?}", target.dim(), target.size());

    let inputs = input.chunk(testratio, 0);
    let targets = target.chunk(testratio, 0);

    let devtype = arg.device.unwrap_or("cpu".to_string());
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
    if let Some(awei) = arg.weight {
        println!("load weight from {}", &awei);
        if let Err(e) = load(
            &awei, &mut vs) {panic!("{e}")}
    }
    let eta = arg.eta;
    let mut optm = nn::AdamW::default().build(&vs, eta)?;
    optm.set_weight_decay(arg.wdecay);
    for (key, t) in vs.variables().iter_mut() {
        println!("{key}:{:?}", t.size());
    }
    let period = arg.anealing;
    let autostop = arg.autostop;
    println!("epoch:{}", arg.epoch);
    println!("eta:{eta}");
    println!("cosine aneaing:{period}");
    println!("mini batch: {}", arg.minibatch);
    println!("weight decay:{}", arg.wdecay);
    println!("test ratio:{testratio}");
    println!("auto stop:{autostop:?}");
    let start = std::time::Instant::now();
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

                let mut dataset = Iter2::new(inp, tar, arg.minibatch);
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
            print!("{}", &epochspeed(ep, arg.epoch, testloss, elapsed));
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
    storeweights(vs);

    Ok(())
}
