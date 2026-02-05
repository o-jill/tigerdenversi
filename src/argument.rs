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
    /// show progressbar
    #[arg(long, default_value_t = false)]
    pub progressbar : bool,
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
    /// parts to train. ex. 1,,0 means only begining part will be trained.
    #[arg(long)]
    pub part : Option<String>,
    /// epochs for warmup sequence.
    #[arg(long, default_value_t = 0)]
    pub warmup : usize,
    /// weight decay for every cos-anealing period
    #[arg(long, default_value_t = 0.001)]
    pub awdecay : f64,
    /// log file path.
    #[arg(long)]
    pub log : Option<String>,
    /// show ascii graph
    #[arg(long, default_value_t = false)]
    pub graph : bool,
    /// get mate(N-1) positions by extracting mateN.
    #[arg(long)]
    pub extract_mate : Option<String>,
    /// ruversi config file
    #[arg(long)]
    pub ru_config : Option<String>,
    /// paths of files which contains mate boards.
    #[arg(long)]
    pub mate_file : Option<String>,
}

impl Arg {
    #[allow(dead_code)]
    //// `--extract-mate`が指定されたかどうか
    /// 棋譜から子供の局面を展開するモードにするか否か
    pub fn is_extract_mate(&self) -> bool {
        self.extract_mate.is_some()
    }

    /// `--extract-mate N`で指定した数字Nを返す。
    /// 指定していないときはゼロを返す。
    pub fn extract_mate_n(&self) -> u32 {
        match self.extract_mate.as_ref() {
            None => {0},
            Some(n) => {
                n.parse().unwrap()
            }
        }
    }
}
