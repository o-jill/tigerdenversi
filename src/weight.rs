use super::*;
use rand::Rng;
use std::{fs, io::{BufReader, BufRead}};

/*
 * input: NUMCELL * NUMCELL + 1(teban) + 2(fixedstones) + 1
 * hidden: 8 + 1
 * output: 1
 */
pub const N_INPUT : usize = bitboard::CELL_2D + 1 + 2;
pub const N_HIDDEN : usize = 32;
pub const N_HIDDEN2 : usize = 16;
const N_OUTPUT : usize = 1;
const N_WEIGHT_TEBAN : usize =  bitboard::CELL_2D * N_HIDDEN;
const N_WEIGHT_FIXST_B : usize = N_WEIGHT_TEBAN + N_HIDDEN;
const N_WEIGHT_FIXST_W : usize = N_WEIGHT_FIXST_B + N_HIDDEN;
const N_WEIGHT_INPUTBIAS : usize = N_WEIGHT_FIXST_W + N_HIDDEN;
const N_WEIGHT_LAYER1 : usize = N_WEIGHT_INPUTBIAS + N_HIDDEN;
const N_WEIGHT_LAYER1BIAS : usize = N_WEIGHT_LAYER1 + N_HIDDEN * N_HIDDEN2;
const N_WEIGHT_LAYER2 : usize = N_WEIGHT_LAYER1BIAS + N_HIDDEN2;
const N_WEIGHT_LAYER2BIAS : usize = N_WEIGHT_LAYER2 + N_HIDDEN2;
const N_WEIGHT : usize =
  (N_INPUT + 1) * N_HIDDEN + (N_HIDDEN + 1) * N_HIDDEN2 + N_HIDDEN2 + 1;


#[allow(dead_code)]
const WSZV1 : usize = (bitboard::CELL_2D + 1 + 1) * 4 + 4 + 1;
#[allow(dead_code)]
const WSZV2 : usize = WSZV1;
const WSZV3 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 4 + 4 + 1;
const WSZV4 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 8 + 8 + 1;
const WSZV5 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 16 + 16 + 1;
const WSZV6 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 32 + 32 + 1;
const WSZV7 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * N_HIDDEN
        + (N_HIDDEN + 1) * N_HIDDEN2 + N_HIDDEN2 + 1;

// v2
// 8/8/1A6/2Ab3/2C3/8/8/8 w
// val:-273.121 val:Some(-273.1215), 268965 nodes. []b6@@b5[]c6@@a7[]a5@@a6[]a8 60msec
// 8/8/1A6/2Ab3/2aB3/1a6/8/8 b
// val:-3.506 val:Some(-3.5055861), 334278 nodes. @@c3[]c2@@d1[]c1@@b1[]a4@@a2 80msec

#[derive(PartialEq)]
pub enum EvalFile{
    Unknown,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
}

impl EvalFile {
    pub fn to_str(&self) -> &str {
        match self {
            EvalFile::Unknown => {"unknown eval file format."},
            EvalFile::V1 => {"# 65-4-1"},
            EvalFile::V2 => {"# 64+1-4-1"},
            EvalFile::V3 => {"# 64+1+2-4-1"},
            EvalFile::V4 => {"# 64+1+2-8-1"},
            EvalFile::V5 => {"# 64+1+2-16-1"},
            EvalFile::V6 => {"# 64+1+2-32-1"},
            EvalFile::V7 => {"# 64+1+2-32-16-1"},
        }
    }

    pub fn from(txt : &str) -> Option<EvalFile> {
        match txt {
            "# 65-4-1" => Some(EvalFile::V1),
            "# 64+1-4-1" => Some(EvalFile::V2),
            "# 64+1+2-4-1" => Some(EvalFile::V3),
            "# 64+1+2-8-1" => Some(EvalFile::V4),
            "# 64+1+2-16-1" => Some(EvalFile::V5),
            "# 64+1+2-32-1" => Some(EvalFile::V6),
            "# 64+1+2-32-16-1" => Some(EvalFile::V7),
            _ => None
        }
    }

    pub fn latest_header() -> String {
        format!("# 64+1+2-{N_HIDDEN}-{N_HIDDEN2}-1")
    }
}

#[repr(align(32))]
pub struct Weight {
    pub weight : [f32 ; N_WEIGHT]
}

impl Weight {
    pub fn new() -> Weight {
        Weight {
            weight: [0.0 ; N_WEIGHT]
        }
    }

    pub fn init(&mut self) {
        let mut rng = rand::thread_rng();
        let range =
            f64::sqrt(6.0) / f64::sqrt((N_INPUT + N_HIDDEN + N_OUTPUT) as f64);

        for a in self.weight.iter_mut() {
            *a = (rng.gen::<f64>() * 2.0 * range - range) as f32;
        }
    }

    #[allow(dead_code)]
    /// fill zero.
    pub fn clear(&mut self) {
        self.weight.iter_mut().for_each(|m| *m = 0.0);
    }

    pub fn wban(&self) -> &[f32] {
        &self.weight[0..]
        // or &self.weight[0..N_WEIGHT_TEBAN]
    }

    pub fn wteban(&self) -> &[f32] {
        &self.weight[N_WEIGHT_TEBAN..N_WEIGHT_FIXST_W]
    }

    pub fn wfixedstones(&self) -> &[f32] {
      &self.weight[N_WEIGHT_FIXST_B..N_WEIGHT_INPUTBIAS]
    }

    pub fn wfixedstone_b(&self) -> &[f32] {
        &self.weight[N_WEIGHT_FIXST_B..N_WEIGHT_FIXST_W]
    }

    pub fn wfixedstone_w(&self) -> &[f32] {
        &self.weight[N_WEIGHT_FIXST_W..N_WEIGHT_INPUTBIAS]
    }

    pub fn wibias(&self) -> &[f32] {
        &self.weight[N_WEIGHT_INPUTBIAS..N_WEIGHT_LAYER1]
    }

    pub fn wlayer1(&self) -> &[f32] {
        &self.weight[N_WEIGHT_LAYER1..N_WEIGHT_LAYER1BIAS]
    }

    pub fn wl1bias(&self) -> &[f32] {
        &self.weight[N_WEIGHT_LAYER1BIAS..N_WEIGHT_LAYER2]
    }

    pub fn wlayer2(&self) -> &[f32] {
        &self.weight[N_WEIGHT_LAYER2..N_WEIGHT_LAYER2BIAS]
    }

    pub fn wl2bias(&self) -> f32 {
        *self.weight.last().unwrap()
    }

    /// read eval table from a file.
    /// 
    /// # Arguments
    /// - `path` file path to a eval table.  
    ///   "RANDOM" is a special text to fill every paramerter w/ random numbers.
    pub fn read(&mut self, path : &str) -> Result<(), String> {
        if path == "RANDOM" {
            self.init();
            return Ok(());
        }
        let mut format = EvalFile::Unknown;
        let file = fs::File::open(path);
        if file.is_err() {return Err(file.err().unwrap().to_string());}

        let file = file.unwrap();
        let lines = BufReader::new(file);
        for line in lines.lines() {
            match line {
                Ok(l) => {
                    if l.starts_with('#') {
                        if format != EvalFile::Unknown {
                            continue;
                        }
                        let res = EvalFile::from(&l);
                        if res.is_some() {
                            format = res.unwrap();
                        }
                        continue;
                    }
                    match format {
                        EvalFile::V1 => {return self.readv1(&l)},
                        EvalFile::V2 => {return self.readv2(&l)},
                        EvalFile::V3 => {return self.readv3(&l)},
                        EvalFile::V4 => {return self.readv4(&l)},
                        EvalFile::V5 => {return self.readv5(&l)},
                        EvalFile::V6 => {return self.readv6(&l)},
                        EvalFile::V7 => {return self.readv7(&l)},
                        _ => {}
                    }
                },
                Err(err) => {return Err(err.to_string())}
            }
        }

        Err("no weight".to_string())
    }

    fn readv1(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v1 format is not supported any more."))
    }

    fn readv2(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v2 format is not supported any more."))
    }

    fn readv3(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v2 format is not supported any more."))
    }

    fn readv4(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v2 format is not supported any more."))
    }

    fn readv5(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v2 format is not supported any more."))
    }

    fn readv6(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(',').collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV6 != nsz {
            return Err(format!("size mismatch v6 {WSZV6} != {nsz}"));
        }
        self.fromv6tov7(&newtable);
        // println!("v6:{:?}", self.weight);
        Ok(())
    }

    fn readv7(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(",").collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV7 != nsz {
            return Err(String::from("size mismatch"));
        }
        self.weight.copy_from_slice(&newtable);
        // println!("v7:{:?}", self.weight);
        Ok(())
    }

    fn write(f : &mut fs::File, w : &[f32], ver : &EvalFile) {
        let sv = w.iter().map(|a| a.to_string()).collect::<Vec<String>>();
        f.write_all(format!("{}\n", ver.to_str()).as_bytes()).unwrap();
        f.write_all(sv.join(",").as_bytes()).unwrap();
    }

    pub fn writev7(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V7);
    }

    #[allow(dead_code)]
    pub fn copy(&mut self, src : &Weight) {
        for (d, s) in self.weight.iter_mut().zip(src.weight.iter()) {
            *d = *s;
        }
    }

    /// copy v3 data into v4.
    fn convert(&mut self, tbl : &[f32], nhid : usize) {
        self.weight = [0.0 ; N_WEIGHT];
        // ban
        let n = nhid * bitboard::CELL_2D;
        let we = &mut self.weight[0..n];
        let tb = &tbl[0..n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // teban
        let idx3 = nhid * bitboard::CELL_2D;
        let idx4 = N_HIDDEN * bitboard::CELL_2D;
        let n = nhid;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // fixed stone
        let idx3 = nhid * (bitboard::CELL_2D + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1);
        let n = nhid;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }
        let idx3 = nhid * (bitboard::CELL_2D + 1 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 1);
        let n = nhid;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // dc
        let idx3 = nhid * (bitboard::CELL_2D + 1 + 2);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2);
        let n = nhid;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // w2
        let idx3 = nhid * (bitboard::CELL_2D + 1 + 2 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2 + 1);
        let n = nhid;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // dc2
        let idx3 = nhid * (bitboard::CELL_2D + 1 + 2 + 1 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2 + 1 + 1);
        self.weight[idx4] =  tbl[idx3];
        // println!("tbl:{tbl:?}");
        // println!("we:{:?}", self.weight);
    }

    /// copy v6 data into v7.
    fn fromv6tov7(&mut self, tbl : &[f32]) {
        self.convert(tbl, 16);
        let mut tmp = [0.0f32 ; N_WEIGHT];
        tmp.copy_from_slice(&self.weight);
        tmp[(N_INPUT + 1)* N_HIDDEN + (N_HIDDEN + 1) * N_HIDDEN2] = 1.0;
        tmp[(N_INPUT + 1)* N_HIDDEN + N_HIDDEN * N_HIDDEN2]
            = tmp[(N_INPUT + 1)* N_HIDDEN + N_HIDDEN];
        tmp[(N_INPUT + 1)* N_HIDDEN + N_HIDDEN] = 0.0;
        self.weight.copy_from_slice(&tmp);
    }

    #[allow(dead_code)]
    pub fn evaluatev3bb(&self, ban : &bitboard::BitBoard) -> f32 {
        let black = ban.black;
        let white = ban.white;
        let teban = ban.teban as f32;
        let ow = &self.weight;

        let fs = ban.fixedstones();

        let mut sum = *ow.last().unwrap();

        let wtbn = &ow[bitboard::CELL_2D * N_HIDDEN .. (bitboard::CELL_2D + 1)* N_HIDDEN];
        let wfs = &ow[(bitboard::CELL_2D + 1) * N_HIDDEN .. (bitboard::CELL_2D + 1 + 2) * N_HIDDEN];
        let wdc = &ow[(bitboard::CELL_2D + 1 + 2) * N_HIDDEN .. (bitboard::CELL_2D + 1 + 2 + 1) * N_HIDDEN];
        let wh = &ow[(bitboard::CELL_2D + 1 + 2 + 1) * N_HIDDEN ..];
        for i in 0..N_HIDDEN {
            let w1 = &ow[i * bitboard::CELL_2D .. (i + 1) * bitboard::CELL_2D];
            let mut hidsum : f32 = 0.0;  // wdc[i];
            for y in 0..bitboard::NUMCELL {
                let mut bit = bitboard::LSB_CELL << y;
                for x in 0..bitboard::NUMCELL {
                    let w = w1[x + y * bitboard::NUMCELL];
                    // let idx = x * bitboard::NUMCELL + y;
                    // let diff = ((bit & black) >> idx) as i32 - ((bit & white) >> idx) as i32;
                    // hidsum += diff as f32 * w;
                    // hidsum += w * ban.at(x, y) as f32;
                    hidsum +=
                        if (bit & black) != 0 {w}
                        else if (bit & white) != 0 {-w}
                        else {0.0};
                    bit <<= bitboard::NUMCELL;
                }
            }
            hidsum = teban.mul_add(wtbn[i], hidsum);
            hidsum = wfs[i].mul_add(fs.0 as f32, hidsum);
            hidsum = wfs[i + N_HIDDEN].mul_add(fs.1 as f32, hidsum + wdc[i]);
            sum = wh[i].mul_add(((-hidsum).exp() + 1.0).recip(), sum);
            // sum += wh[i] / ((-hidsum).exp() + 1.0);
        }
        sum
    }
}
