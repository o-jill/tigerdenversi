use std::io::Write;

use clap::Parser;
use tch::{nn, nn::{Module, OptimizerConfig, VarStore}};
use tch::{Device, data::Iter2, Kind, Tensor};

mod kifu;
mod bitboard;
mod weight;

const INPUTSIZE :i64 = 8 * 8 + 1 + 2;
const HIDDENSIZE : i64 = 32;
// const MINIBATCH : i64 = 16;
const MIN_COSANEAL : f64 = 1e-3;

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
    device : Option<String>
}

fn net(vs : &nn::Path) -> impl Module {
    nn::seq()
        .add(
            nn::linear(vs / "layer1",
            INPUTSIZE, HIDDENSIZE, Default::default()))
            // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(vs / "layer2", HIDDENSIZE, 1, Default::default()))
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

fn loadkifu(files : &[String]) -> Vec<(bitboard::BitBoard, i8, i8, i8, i8)> {
    let mut boards : Vec<(bitboard::BitBoard, i8, i8, i8, i8)> = Vec::new();
    for fname in files.iter() {
        let path = format!("kifu/{}", fname);
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
            boards.push((ban, t.teban, fsb, fsw, kifu.score.unwrap()));
            boards.push((ban90, t.teban, fsb, fsw, kifu.score.unwrap()));
            boards.push((ban180, t.teban, fsb, fsw, kifu.score.unwrap()));
            boards.push((ban270, t.teban, fsb, fsw, kifu.score.unwrap()));
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

// load from `fname` into `vs`.
//
// .safetensor and ruversi weight format are available.
fn load(fname : &str, vs : &mut VarStore) -> Result<(), String> {
    let path = std::path::Path::new(fname);
    if !path.exists() {
        return Err(format!("{fname} was not found..."));
    }
    if path.ends_with(".safetensors") {
        println!("load weight: {}", fname);
        match vs.load(path) {
            Err(e) => {return Err(e.to_string())},
            _ => {return Ok(());}
        }
    }

    let mut txtweight = weight::Weight::new();
    txtweight.read(fname)?;

    const INPSIZE : usize = bitboard::CELL_2D + 1 + 2;
    const HIDSIZE : usize =  HIDDENSIZE as usize;
    let wban = &txtweight.weight;
    let wtbn = &wban[bitboard::CELL_2D * HIDSIZE..];
    let wfs = &wban[(bitboard::CELL_2D + 1) * HIDSIZE..];
    let wdc = &wban[INPSIZE * HIDSIZE..(INPSIZE + 1) * HIDSIZE];
    let whdn = &wban[(INPSIZE + 1) * HIDSIZE..(INPSIZE + 1 + 1) * HIDSIZE];
    let wdc2 = wban.last().unwrap();

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
    {
        let mut val = vs.variables_.lock();
        val.as_mut().unwrap().named_variables.insert(
            "layer1.weight".to_string(), wl1);
    }

    // layer1.bias
    let mut bias = [0.0f32 ; HIDDENSIZE as usize];
    bias.copy_from_slice(wdc);
    let wb1 = Tensor::from_slice(&bias).view((1, HIDDENSIZE));
    {
        let mut val = vs.variables_.lock();
        val.as_mut().unwrap().named_variables.insert(
            "layer1.bias".to_string(), wb1);
    }

    // layer2.weight
    let mut weights = [0.0f32 ; HIDDENSIZE as usize];
    weights.copy_from_slice(whdn);
    let wl2 = Tensor::from_slice(&weights).view(HIDDENSIZE);
    {
        let mut val = vs.variables_.lock();
        val.as_mut().unwrap().named_variables.insert(
            "layer2.weight".to_string(), wl2);
    }

    // layer2.bias
    let mut bias = [0.0f32 ; 1];
    bias[0] = *wdc2;
    let wb2 = Tensor::from_slice(&bias).view(1);
    {
        let mut val = vs.variables_.lock();
        val.as_mut().unwrap().named_variables.insert(
            "layer2.bias".to_string(), wb2);
    }
    Ok(())
}

fn storeweights(vs : VarStore) {
    println!("save to weight.safetensors");
    vs.save("weight.safetensors").unwrap();

    // VarStore to weights
    let weights = vs.variables();
    let mut outp = [0.0f32 ; (INPUTSIZE * HIDDENSIZE) as usize];
    let mut params = format!("# 64+1+2-{HIDDENSIZE}-1\n");
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
    let keys = ["layer1.bias", "layer2.weight"];
    for key in keys {
        let l1w = weights.get(key).unwrap();
        println!("{key}:{:?}", l1w.size());
        let numel = l1w.numel();
        l1w.copy_data(outp.as_mut_slice(), numel);
        params += &outp[0..numel].iter()
            .map(|a| format!("{a}")).collect::<Vec<String>>().join(",");
        params += ",";
    }
    let l1w = weights.get("layer2.bias").unwrap();
    println!("layer2.bias:{:?}", l1w.size());
    let numel = l1w.numel();
    l1w.copy_data(outp.as_mut_slice(), numel);
    params += &outp[0..numel].iter()
        .map(|a| format!("{a}")).collect::<Vec<String>>().join(",");
    println!("save to weight.txt");
    let mut f = std::fs::File::create("weights.txt").unwrap();
    f.write_all(params.as_bytes()).unwrap();
}

fn epochspeed(
    ep : usize, maxepoch : usize, elapsed : std::time::Duration) -> String {
    let epoch = ep + 1;
    let speed = elapsed.as_secs_f64() / (epoch) as f64;

    let etasecs = (maxepoch - epoch) as f64 * speed;

    let esthour = (etasecs / 3600.0) as i32;
    let estmin = ((etasecs - esthour as f64 * 3600.0) / 60.0) as i32;
    let estsec = (etasecs % 60.0) as i32;

    let mut res = format!("ep:{epoch}/{maxepoch} ");
    res += &format!("ETA:{esthour:02}h{estmin:02}m{estsec:02}s ");
    res + &if speed > 3600.0 {
            format!("{:.1}hour/epoch\r", speed / 3600.0)
        } else  if speed > 60.0 {
            format!("{:.1}min/epoch\r", speed / 60.0)
        } else {
            format!("{speed:.1}sec/epoch\r")
        }
}

fn main() -> Result<(), tch::TchError> {
    let t = Tensor::f_rand([1, 8, 8], (Kind::Float, Device::Cpu))?;
    // let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    let arg = Arg::parse();

    let kifupath = "./kifu";
    let mut boards = loadkifu(&findfiles(kifupath));
    dedupboards(&mut boards);

    let input = tch::Tensor::from_slice(
        &extractboards(&boards)).view((boards.len() as i64, INPUTSIZE));
    println!("input : {} {:?}", input.dim(), input.size());

    let target = tch::Tensor::from_slice(
        &extractscore(&boards)).view((boards.len() as i64, 1));
    println!("target: {} {:?}", target.dim(), target.size());

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
    if arg.weight.is_some() {
        println!("load weight: {}", arg.weight.as_ref().unwrap());
        if let Err(e) = load(
            arg.weight.as_ref().unwrap(), &mut vs) {panic!("{e}")}
    }
    let eta = arg.eta;
    let mut optm = nn::AdamW::default().build(&vs, eta)?;
    for (key, t) in vs.variables().iter_mut() {
        println!("{key}:{:?}", t.size());
    }
    let period = arg.anealing;
    println!("epoch:{}", arg.epoch);
    println!("eta:{eta}");
    println!("cosine aneaing:{period}");
    println!("mini batch: {}", arg.minibatch);
    let start = std::time::Instant::now();
    if period > 1 {
        for ep in 0..arg.epoch {
            optm.set_lr(
                eta * MIN_COSANEAL +
                    eta * 0.5 * (1.0 - MIN_COSANEAL)
                        * (1.0 + (ep as f64 / period as f64).cos())
            );
            let mut dataset = Iter2::new(&input, &target, arg.minibatch);
            let dataset = dataset.shuffle();
            // let mut loss = tch::Tensor::new();
            for (xs, ys) in dataset {
                // println!("xs: {} {:?} ys: {} {:?}",
                //          xs.dim(), xs.size(), ys.dim(), ys.size());
                let loss =
                    nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                optm.backward_step(&loss);
            }
            // let accu = nnet.batch_accuracy_for_logits(
            //         &input, &target, vs.device(), 400);
            // println!("ep:{ep}, {}, {:.3}",
            //          loss.sum(Some(tch::Kind::Float)), accu * 100.00);
            let elapsed = start.elapsed();
            print!("{}", &epochspeed(ep, arg.epoch, elapsed));
            std::io::stdout().flush().unwrap();
        }
    } else {
        for ep in 0..arg.epoch {
            let mut dataset = Iter2::new(&input, &target, arg.minibatch);
            let dataset = dataset.shuffle().to_device(vs.device());
            // let mut loss = tch::Tensor::new();
            for (xs, ys) in dataset {
                // println!("xs: {} {:?} ys: {} {:?}",
                //          xs.dim(), xs.size(), ys.dim(), ys.size());
                let loss =
                    nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
                optm.backward_step(&loss);
            }
            // let accu = nnet.batch_accuracy_for_logits(
            //         &input, &target, vs.device(), 400);
            // println!("ep:{ep}, {}, {:.3}",
            //          loss.sum(Some(tch::Kind::Float)), accu * 100.00);
            let elapsed = start.elapsed();
            print!("{}", &epochspeed(ep, arg.epoch, elapsed));
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    // VarStore to weights
    storeweights(vs);

    Ok(())
}
