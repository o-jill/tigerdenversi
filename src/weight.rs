use super::*;
use rand::Rng;
use std::{fs, io::{BufReader, BufRead}};

/*
 * input: NUMCELL * NUMCELL + 1(teban) + 2(fixedstones) + 1
 * hidden: 8 + 1
 * output: 1
 */
const N_INPUT : usize = bitboard::CELL_2D + 1 + 2;
const N_HIDDEN : usize = 32;
const N_OUTPUT : usize = 1;
pub const N_WEIGHT: usize = (N_INPUT + 1) * N_HIDDEN + N_HIDDEN + 1;

#[allow(dead_code)]
const WSZV1 : usize = (bitboard::CELL_2D + 1 + 1) * 4 + 4 + 1;
#[allow(dead_code)]
const WSZV2 : usize = WSZV1;
const WSZV3 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 4 + 4 + 1;
const WSZV4 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 8 + 8 + 1;
const WSZV5 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 16 + 16 + 1;
const WSZV6 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * N_HIDDEN + N_HIDDEN + 1;

// v2
// 8/8/1A6/2Ab3/2C3/8/8/8 w
// val:-273.121 val:Some(-273.1215), 268965 nodes. []b6@@b5[]c6@@a7[]a5@@a6[]a8 60msec
// 8/8/1A6/2Ab3/2aB3/1a6/8/8 b
// val:-3.506 val:Some(-3.5055861), 334278 nodes. @@c3[]c2@@d1[]c1@@b1[]a4@@a2 80msec

#[derive(PartialEq)]
enum EvalFile{
    Unknown,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
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
            _ => None
        }
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

    fn readv3(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(',').collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV3 != nsz {
            return Err(format!("size mismatch {WSZV3} != {nsz}"));
        }
        self.fromv3tov6(&newtable);
        // println!("v3:{:?}", self.weight);
        Ok(())
    }

    fn readv4(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(',').collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV4 != nsz {
            return Err(String::from("size mismatch v4"));
        }
        self.fromv4tov6(&newtable);
        // println!("v4:{:?}", self.weight);
        Ok(())
    }

    fn readv5(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(',').collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV5 != nsz {
            return Err(String::from("size mismatch v5"));
        }
        self.fromv5tov6(&newtable);
        // println!("v5:{:?}", self.weight);
        Ok(())
    }

    fn readv6(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(',').collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV6 != nsz {
            return Err(format!("size mismatch v6 {WSZV6} != {nsz}"));
        }
        self.weight.copy_from_slice(&newtable);
        // println!("v6:{:?}", self.weight);
        Ok(())
    }

    fn write(f : &mut fs::File, w : &[f32], ver : &EvalFile) {
        let sv = w.iter().map(|a| a.to_string()).collect::<Vec<String>>();
        f.write_all(format!("{}\n", ver.to_str()).as_bytes()).unwrap();
        f.write_all(sv.join(",").as_bytes()).unwrap();
    }

    #[allow(dead_code)]
    pub fn writev1(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V1);
    }

    #[allow(dead_code)]
    pub fn writev2(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V2);
    }

    #[allow(dead_code)]
    pub fn writev3(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V3);
    }

    #[allow(dead_code)]
    pub fn writev4(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V4);
    }

    #[allow(dead_code)]
    pub fn writev5(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V5);
    }

    #[allow(dead_code)]
    pub fn writev6(&self, path : &str) {
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V6);
    }

    #[allow(dead_code)]
    pub fn writev1asv2(&self, path : &str) {
        let mut w = Weight::new();
        w.fromv1tov2(&self.weight);
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V2);
    }

    #[allow(dead_code)]
    pub fn writev2asv3(&self, path : &str) {
        let mut w = Weight::new();
        w.fromv2tov3(&self.weight);
        let mut f = fs::File::create(path).unwrap();
        Weight::write(&mut f, &self.weight, &EvalFile::V2);
    }

    #[allow(dead_code)]
    pub fn copy(&mut self, src : &Weight) {
        for (d, s) in self.weight.iter_mut().zip(src.weight.iter()) {
            *d = *s;
        }
    }

    fn fromv1tov2(&mut self, tbl : &[f32]) {
        // ban
        for i in 0..N_HIDDEN {
            let we = &mut self.weight[i * bitboard::CELL_2D..(i + 1) * bitboard::CELL_2D];
            let tb = &tbl[i * (bitboard::CELL_2D + 1 + 1)..(i + 1) * (bitboard::CELL_2D + 1 + 1)];
            for (w, t) in we.iter_mut().zip(tb.iter()) {
                *w = *t;
            }
            let teb = &mut self.weight[
                N_HIDDEN * bitboard::CELL_2D + i..=N_HIDDEN * bitboard::CELL_2D + N_HIDDEN * 2 + i];
            // teban
            teb[0] = tbl[i * (bitboard::CELL_2D + 1 + 1) + bitboard::CELL_2D];
            // dc
            teb[N_HIDDEN] = tbl[i * (bitboard::CELL_2D + 1 + 1) + bitboard::CELL_2D + 1];
            // hidden
            teb[N_HIDDEN * 2] = tbl[4 * (bitboard::CELL_2D + 1 + 1) + i];
        }
        // dc
        *self.weight.last_mut().unwrap() = *tbl.last().unwrap();
    }

    #[allow(dead_code)]
    fn fromv1tov3(&mut self, tbl : &[f32]) {
        // ban
        for i in 0..N_HIDDEN {
            let we = &mut self.weight[i * bitboard::CELL_2D..(i + 1) * bitboard::CELL_2D];
            let tb = &tbl[i * (bitboard::CELL_2D + 1 + 1)..(i + 1) * (bitboard::CELL_2D + 1 + 1)];
            for (w, t) in we.iter_mut().zip(tb.iter()) {
                *w = *t;
            }
            let teb = &mut self.weight[N_HIDDEN * bitboard::CELL_2D + i..];
            // teban
            teb[0] = tbl[i * (bitboard::CELL_2D + 1 + 1) + bitboard::CELL_2D];
            // fixed stone
            teb[N_HIDDEN] = 0.0;
            teb[N_HIDDEN * 2] = 0.0;
            // dc
            teb[N_HIDDEN * 3] = tbl[i * (bitboard::CELL_2D + 1 + 1) + bitboard::CELL_2D + 1];
            // hidden
            teb[N_HIDDEN * 4] = tbl[4 * (bitboard::CELL_2D + 1 + 1) + i];
        }
        // dc
        *self.weight.last_mut().unwrap() = *tbl.last().unwrap();
    }

    fn fromv2tov3(&mut self, tbl : &[f32]) {
        // ban + teban
        let we = &mut self.weight[0..N_HIDDEN * (bitboard::CELL_2D + 1)];
        let tb = &tbl[0 .. N_HIDDEN * (bitboard::CELL_2D + 1)];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // fixed stone
        let we = &mut self.weight[N_HIDDEN * (bitboard::CELL_2D + 1) .. N_HIDDEN * (bitboard::CELL_2D + 1 + 2)];
        we.fill(0.0);

        // dc + w2 + dc2
        let we = &mut self.weight[N_HIDDEN * (bitboard::CELL_2D + 1 + 2)..];
        let dcw2 = &tbl[N_HIDDEN * (bitboard::CELL_2D + 1)..];
        for (w, t) in we.iter_mut().zip(dcw2.iter()) {
            *w = *t;
        }
    }

    /// copy v3 data into v4.
    #[allow(dead_code)]
    fn fromv3tov4(&mut self, tbl : &[f32]) {
        self.weight = [0.0 ; N_WEIGHT];
        // ban
        let n = 4 * bitboard::CELL_2D;
        let we = &mut self.weight[0..n];
        let tb = &tbl[0..n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // teban
        let idx3 = 4 * bitboard::CELL_2D;
        let idx4 = N_HIDDEN * bitboard::CELL_2D;
        let n = 4;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // fixed stone
        let idx3 = 4 * (bitboard::CELL_2D + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1);
        let n = 4;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }
        let idx3 = 4 * (bitboard::CELL_2D + 1 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 1);
        let n = 4;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // dc
        let idx3 = 4 * (bitboard::CELL_2D + 1 + 2);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2);
        let n = 4;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // w2
        let idx3 = 4 * (bitboard::CELL_2D + 1 + 2 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2 + 1);
        let n = 4;
        let we = &mut self.weight[idx4..idx4 + n];
        let tb = &tbl[idx3..idx3 + n];
        for (w, t) in we.iter_mut().zip(tb.iter()) {
            *w = *t;
        }

        // dc2
        let idx3 = 4 * (bitboard::CELL_2D + 1 + 2 + 1 + 1);
        let idx4 = N_HIDDEN * (bitboard::CELL_2D + 1 + 2 + 1 + 1);
        self.weight[idx4] =  tbl[idx3];
        // println!("tbl:{tbl:?}");
        // println!("we:{:?}", self.weight);
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

    /// copy v3 data into v6.
    fn fromv3tov6(&mut self, tbl : &[f32]) {
        self.convert(tbl, 4);
    }

    /// copy v4 data into v6.
    fn fromv4tov6(&mut self, tbl : &[f32]) {
        self.convert(tbl, 8);
    }

    /// copy v5 data into v6.
    fn fromv5tov6(&mut self, tbl : &[f32]) {
        self.convert(tbl, 16);
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
