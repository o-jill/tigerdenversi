use super::*;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, author, about)]
pub struct Arg {
    /// path for weight.safetensor to use
    #[arg(short, long)]
    pub weight : Option<String>,
    /// initial learning rate
    #[arg(short, long, default_value_t = 0.01)]
    pub eta : f64,
    /// # of epochs
    #[arg(long, default_value_t = 100)]
    pub epoch : usize,
    /// kifu directory
    #[arg(long)]
    pub kifudir : Option<String>,
    /// mini batch size
    #[arg(long, default_value_t = 16)]
    pub minibatch : i64,
    /// storing weight after some iterations as weight.EPOCH.txt.
    #[arg(short, long)]
    pub progress : Option<String>,
    /// cosine anealing period.
    #[arg(short, long, default_value_t = 0)]
    pub anealing : i32,
    /// device to process. cuda, mps or cpu. default:cpu.
    #[arg(long)]
    pub device : Option<String>,
    /// weight decay
    #[arg(long, default_value_t = 0.0002)]
    pub wdecay : f64,
    /// ratio of test data for calc loss
    #[arg(long, default_value_t = 5)]
    pub testratio : usize,
    /// check if trained enough. [prefered: 0]
    #[arg(long)]
    #[structopt(allow_hyphen_values = true)]
    pub autostop : Option<f64>,
    /// parts to train. ex. 1,,0 means ending part.
    #[arg(long)]
    pub part : Option<String>,
    /// epochs for warmup sequence.
    #[arg(long, default_value_t = 0)]
    pub warmup : usize,
}

impl Arg {
    /// csv text to get an array if each part will be trained or not.
    /// "", "0", "false", "no", "none", "off" and "zero" disables training.
    ///
    /// ex. "" becomes [true, true, true]
    /// ex. "1,,0" becomes [true, false, false]
    /// ex. "-1,false,zero" becomes [true, false, false]
    pub fn partlist(&self) -> Vec<bool> {
        let mut ret = vec![true ; weight::N_PROGRESS_DIV];
        if self.part.is_none() {
            return ret;
        }

        let txt = self.part.as_ref().unwrap();
        let disable = ["", "0", "false", "no", "none", "off", "zero"];
        let txt_lo = txt.to_lowercase();
        for (i, elem) in txt_lo.split(',').enumerate() {
            if i >= weight::N_PROGRESS_DIV {break;}

            ret[i] = disable.contains(&elem);
        }
        ret
    }
}