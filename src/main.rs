use std::default;
use std::io::Write;

use bitboard::CELL_2D;
use tch::nn::{LinearConfig, ModuleT, OptimizerConfig};
use tch::{data, Kind, Tensor};
use tch::{nn, {nn::VarStore, nn::Module}};
use tch::Device;
use tch::data::Iter2;

mod kifu;
mod bitboard;

const INPUTSIZE :i64 = 8 * 8 + 1 + 2;
const HIDDENSIZE : i64 = 16;
const MINIBATCH : i64 = 16;

fn net(vs : &nn::Path) -> impl Module {
    nn::seq()
        .add(
            nn::linear(vs / "layer1",
            INPUTSIZE, HIDDENSIZE, Default::default()))
            // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
        .add_fn(|xs| xs.sigmoid())
        .add(nn::linear(vs / "layer2", HIDDENSIZE, 1, Default::default()))
}

fn main() -> Result<(), tch::TchError> {
    let t = Tensor::f_rand([1, 8, 8], (Kind::Float, Device::Cpu))?;
    // let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    let kifupath = "./kifu";
    // list up kifu
    let dir = std::fs::read_dir(kifupath).unwrap();
    let mut files = dir.filter_map(|entry| {
        entry.ok().and_then(|e|
            e.path().file_name().and_then(|n|
                Some(n.to_str().unwrap().to_string())
            )
        )}).collect::<Vec<String>>().iter().filter(|&fnm| {
            fnm.contains("kifu")
            // fnm.contains(".txt")
        }).cloned().collect::<Vec<String>>();
    // println!("{:?}", files);

    files.sort();
    let mut boards : Vec<(bitboard::BitBoard, i8, i8, i8, i8)> = Vec::new();
    for fname in files.iter() {
        let path = format!("kifu/{}", fname);
        println!("{path}");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines:Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        for t in kifu.list.iter() {
            let ban = bitboard::BitBoard::from(&t.rfen).unwrap();
            let (fsb, fsw) = ban.fixedstones();
            boards.push((ban, t.teban, fsb, fsw, kifu.score.unwrap()));
        }
    }
    println!("board: {} boards", boards.len());
    let inputlist = boards.iter().map(|(b, t, fb, fw, _s)| {
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
    }).collect::<Vec<[f32 ; INPUTSIZE as usize]>>().concat();
    let input = tch::Tensor::from_slice(
        &inputlist).view((boards.len() as i64, INPUTSIZE));
    println!("input : {} {:?}", input.dim(), input.size());
    let targetlist = boards.iter()
        .map(|(_b, _t, _fb, _fw, s)| *s as f32).collect::<Vec<f32>>();
    let target = tch::Tensor::from_slice(
        &targetlist).view((boards.len() as i64, 1));
    println!("target: {} {:?}", target.dim(), target.size());

    let vs = VarStore::new(Device::Cpu);
    let nnet = net(&vs.root());
    let mut optm = nn::Adam::default().build(&vs, 1e-4)?;
    for (key, t) in vs.variables().iter_mut() {
        println!("{key}:{:?}", t.size());
    }

    for epock in 1..20 {
        let mut dataset = data::Iter2::new(&input, &target, MINIBATCH);
        let dataset = dataset.shuffle();
        let mut loss = tch::Tensor::new();
        for (xs, ys) in dataset {
            // println!("xs: {} {:?} ys: {} {:?}", xs.dim(), xs.size(), ys.dim(), ys.size());
            loss = nnet.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
            optm.backward_step(&loss);
        }
        let accu = nnet.batch_accuracy_for_logits(&input, &target, vs.device(), 400);
        println!("ep:{epock}, {}, {:.3}", loss.sum(Some(tch::Kind::Float)), accu * 10.00);
    }
    vs.save("weight.safetensors").unwrap();

    // VarStore to weights
    let weights = vs.variables();
    let mut outp = [0.0f32 ; (INPUTSIZE * HIDDENSIZE) as usize];
    let mut params = String::new();
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
    let keys = ["layer1.bias", "layer2.weight", "layer2.bias"];
    for key in keys {
        let l1w = weights.get(key).unwrap();
        println!("{key}:{:?}", l1w.size());
        let numel = l1w.numel();
        l1w.copy_data(outp.as_mut_slice(), numel);
        params += &outp[0..numel].iter()
            .map(|a| format!("{a},")).collect::<Vec<String>>().join("");
    }
    let mut f = std::fs::File::create("weights.txt").unwrap();
    f.write_all(params.as_bytes()).unwrap();

    Ok(())
}