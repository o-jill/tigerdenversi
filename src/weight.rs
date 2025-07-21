use super::*;
use rand::Rng;
use std::{fs, io::{BufReader, BufRead}};

/*
 * input: NUMCELL * NUMCELL + 1(teban) + 2(fixedstones) + 1
 * hidden: 8 + 1
 * output: 1
 */
pub const N_INPUT : usize = bitboard::CELL_2D + 1 + 2;
pub const N_HIDDEN : usize = 128;
pub const N_HIDDEN2 : usize = 16;
const N_OUTPUT : usize = 1;
pub const N_WEIGHT_TEBAN : usize =  bitboard::CELL_2D * N_HIDDEN;
pub const N_WEIGHT_FIXST_B : usize = N_WEIGHT_TEBAN + N_HIDDEN;
pub const N_WEIGHT_FIXST_W : usize = N_WEIGHT_FIXST_B + N_HIDDEN;
pub const N_WEIGHT_INPUTBIAS : usize = N_WEIGHT_FIXST_W + N_HIDDEN;
pub const N_WEIGHT_LAYER1 : usize = N_WEIGHT_INPUTBIAS + N_HIDDEN;
pub const N_WEIGHT_LAYER1BIAS : usize = N_WEIGHT_LAYER1 + N_HIDDEN * N_HIDDEN2;
pub const N_WEIGHT_LAYER2 : usize = N_WEIGHT_LAYER1BIAS + N_HIDDEN2;
const N_WEIGHT_LAYER2BIAS : usize = N_WEIGHT_LAYER2 + N_HIDDEN2;
pub const N_WEIGHT : usize =
  (N_INPUT + 1) * N_HIDDEN + (N_HIDDEN + 1) * N_HIDDEN2 + N_HIDDEN2 + 1;

const N_WEIGHT_PAD :usize = ((N_WEIGHT + 7) / 8) * 8;
pub const N_PROGRESS_DIV : usize = 3;

#[allow(dead_code)]
const WSZV1 : usize = (bitboard::CELL_2D + 1 + 1) * 4 + 4 + 1;
#[allow(dead_code)]
const WSZV2 : usize = WSZV1;
const WSZV3 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 4 + 4 + 1;
const WSZV4 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 8 + 8 + 1;
const WSZV5 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 16 + 16 + 1;
const WSZV6 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 32 + 32 + 1;
const WSZV7 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * 32
        + (32 + 1) * 16 + 16 + 1;
// ^^^^^ sigmoid
// vvvvv relu
const WSZV8 : usize = (bitboard::CELL_2D + 1 + 2 + 1) * N_HIDDEN
        + (N_HIDDEN + 1) * N_HIDDEN2 + N_HIDDEN2 + 1;
const WSZV9 : usize = WSZV8;

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
    V8,
    V9,
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
            EvalFile::V8 => {"# 64+1+2-128-16-1"},
            EvalFile::V9 => {"# 3x 64+1+2-128-16-1"},
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
            "# 64+1+2-128-16-1" => Some(EvalFile::V8),
            "# 3x 64+1+2-128-16-1" => Some(EvalFile::V9),
            _ => None
        }
    }

    pub fn latest_header() -> String {
        format!("# x{N_PROGRESS_DIV} 64+1+2-{N_HIDDEN}-{N_HIDDEN2}-1")
    }
}

#[repr(align(32))]
pub struct Weight {
    pub weight : [f32 ; N_WEIGHT_PAD * N_PROGRESS_DIV],
}

impl Default for Weight {
    fn default() -> Self {
        let mut w = Self::new();
        w.init();
        w
    }
}

impl Weight {
    pub fn new() -> Weight {
        Weight {
            weight: [0.0 ; N_WEIGHT_PAD * N_PROGRESS_DIV]
        }
    }

    pub fn init(&mut self) {
        let mut rng = rand::thread_rng();
        let range =
            f64::sqrt(6.0) /
                f64::sqrt((N_INPUT + N_HIDDEN + N_HIDDEN2 + N_OUTPUT) as f64);

        for a in self.weight.iter_mut() {
            *a = (rng.gen::<f64>() * 2.0 * range - range) as f32;
        }
    }

    /// copy weights from array
    pub fn copy_from_slice(&mut self, array : &[f32], progress : usize) {
        let offset = progress * N_WEIGHT_PAD;
        self.weight[offset..offset + N_WEIGHT].copy_from_slice(array);
    }

    #[allow(dead_code)]
    /// fill zero.
    pub fn clear(&mut self) {
        self.weight.iter_mut().for_each(|m| *m = 0.0);
    }

    pub fn wban(&self, progress : usize) -> &[f32] {
        &self.weight[progress * N_WEIGHT_PAD..]
    }

    pub fn wteban(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_TEBAN..offset + N_WEIGHT_FIXST_W]
    }

    pub fn wfixedstones(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
      &self.weight[offset + N_WEIGHT_FIXST_B..offset + N_WEIGHT_INPUTBIAS]
    }

    pub fn wfixedstone_b(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_FIXST_B..offset + N_WEIGHT_FIXST_W]
    }

    pub fn wfixedstone_w(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_FIXST_W..offset + N_WEIGHT_INPUTBIAS]
    }

    pub fn wibias(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_INPUTBIAS..offset + N_WEIGHT_LAYER1]
    }

    pub fn wlayer1(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_LAYER1..offset + N_WEIGHT_LAYER1BIAS]
    }

    pub fn wl1bias(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_LAYER1BIAS..offset + N_WEIGHT_LAYER2]
    }

    pub fn wlayer2(&self, progress : usize) -> &[f32] {
        let offset = progress * N_WEIGHT_PAD;
        &self.weight[offset + N_WEIGHT_LAYER2..offset + N_WEIGHT_LAYER2BIAS]
    }

    pub fn wl2bias(&self, progress : usize) -> f32 {
        let offset = progress * N_WEIGHT_PAD;
        self.weight[offset + N_WEIGHT - 1]
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

        let mut idx = 0;
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
                        EvalFile::V8 => {return self.readv8(&l)},
                        EvalFile::V9 => {
                            self.readv9(&l, idx)?;
                            idx += 1;
                            if idx >= N_PROGRESS_DIV {return Ok(());}
                        },
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

    fn readv6(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v1 format is not supported any more."))
    }

    fn readv7(&mut self, _line : &str) -> Result<(), String> {
        Err(String::from("v1 format is not supported any more."))
    }

    fn readv8(&mut self, line : &str) -> Result<(), String> {
        let csv = line.split(",").collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV8 != nsz {
            return Err(format!("size mismatch v9:{WSZV9} != {nsz}"));
        }

        for prgs in 0..N_PROGRESS_DIV {
            let offset = prgs * N_WEIGHT_PAD;
            self.weight[offset..offset + N_WEIGHT].copy_from_slice(&newtable);
        }
        // println!("v8:{:?}", self.weight);
        Ok(())
    }

    fn readv9(&mut self, line : &str, progress : usize) -> Result<(), String> {
        let csv = line.split(",").collect::<Vec<_>>();
        let newtable : Vec<f32> = csv.iter().map(|&a| a.parse::<f32>().unwrap()).collect();
        let nsz = newtable.len();
        if WSZV9 != nsz {
            return Err(String::from("size mismatch"));
        }
        self.copy_from_slice(&newtable, progress);
        // println!("v8:{:?}", self.weight);
        Ok(())
    }

    pub fn writev9(&self, path : &str) ->Result<(), std::io::Error> {
        // header
        let mut outp = format!("{}\n", EvalFile::V9.to_str());

        // weights
        for prgs in 0..N_PROGRESS_DIV {
            let offset = prgs * N_WEIGHT_PAD;
            let w = &self.weight[offset..offset + N_WEIGHT];
            let sv = w.iter().map(|a| a.to_string()).collect::<Vec<String>>();
            outp += &sv.join(",");
            outp += "\n";
        }

        // put to a file
        let mut f = fs::File::create(path).unwrap();
        f.write_all(outp.as_bytes())?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn copy(&mut self, src : &Weight) {
        for (d, s) in self.weight.iter_mut().zip(src.weight.iter()) {
            *d = *s;
        }
    }
}
