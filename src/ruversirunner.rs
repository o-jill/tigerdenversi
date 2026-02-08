use std::io::{BufReader, BufRead, Write};
use std::process::{Child, Command, Stdio};

use crate::bitboard;

/**
 * configファイルのargsタグを処理する。
 * # Argumemts  
 * - txt args:を含まない"arg:"の後ろに書いてある文字列。
 * # Returns  
 * - Ok(Vec<String>)  
 *   引数に渡したい文字列の配列
 * - Err(String>)  
 *   処理エラーの内容
 */
fn parse_args_tag(txt : &str) -> Result<Vec<String>, String> {
    let args = txt.trim().split(",")
        .map(|s| s.trim().to_string()).collect::<Vec<_>>();
    if args.len() > 1 || !args[0].is_empty() {
        for (i, a) in args.iter().enumerate() {
            if a.is_empty() {
                return Err(
                    format!("\"{txt}\" contains empty part @{i}! {args:?}"));
            }
        }
        Ok(args)
    } else  {
        Ok(Vec::new())
    }
}

/// run ruversi
pub struct RuversiRunner {
    curdir : String,
    path : String,
    evfile : String,
    verbose : bool,
    args : Vec<String>,
}

impl std::fmt::Display for RuversiRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "curdir:{}, ruversi:{}, evfile:{}, args:{:?}",
               self.curdir, self.path, self.evfile, self.args)
    }
}

impl RuversiRunner {
    pub fn new() -> RuversiRunner {
        RuversiRunner {
            curdir : String::from("../ruversi"),
            path : String::from("./target/release/ruversi"),
            evfile : String::from("data/evaltable.txt"),
            verbose : false,
            args : vec!["--thinkall".to_string()],
        }
    }

    pub fn set_verbose(&mut self, verbose : bool) {
        self.verbose = verbose;
    }

    pub fn from_config(path : &std::path::PathBuf)
            -> Result<RuversiRunner, String> {
        let mut rr = RuversiRunner::new();
        if path.as_os_str().is_empty() {
            return Ok(rr);
        }

        match rr.read(path) {
            Ok(_) => Ok(rr),
            Err(msg) => Err(msg),
        }
    }

    /// read config from a file.
    /// 
    /// ex.
    /// curdir: ~/ruversi/
    /// path: ./bin/ruversi
    /// evfile: ./data/eval.dat
    /// args: --depth,7,--silent
    /// 
    /// note:
    /// COMMAs between `"`s are not be ignored yet...
    /// "a,b" will be `"a` and `b"`...
    pub fn read(&mut self, path : &std::path::PathBuf) -> Result<(), String> {
        let file = std::fs::File::open(path);
        if file.is_err() {return Err(file.err().unwrap().to_string());}

        let file = file.unwrap();
        let lines = BufReader::new(file);
        for line in lines.lines() {
            match line {
                Ok(l) => {
                    if let Some(cd) = l.strip_prefix("curdir:") {
                        self.curdir = String::from(cd.trim());
                    } else if let Some(ed) = l.strip_prefix("path:") {
                        self.path = String::from(ed.trim());
                    } else if let Some(evf) = l.strip_prefix("evfile:") {
                        self.evfile = String::from(evf.trim());
                    } else if let Some(args_txt) = l.strip_prefix("args:") {
                        self.args = parse_args_tag(&args_txt)?;
                    }
                },
                Err(err) => {return Err(err.to_string())}
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn spawn(&self, rfen : &str) -> std::io::Result<Child> {
        std::env::set_current_dir(&self.curdir).unwrap();
        let mut cmd = Command::new(&self.path);
        cmd.arg("--rfen").arg(rfen).arg("--ev1").arg(&self.evfile)
            .args(&self.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if self.verbose {eprintln!("cmd:{cmd:?}");}
        cmd.spawn()
    }

    fn spawn_children(&self, rfen : &str, depth : u32) -> std::io::Result<Child> {
        std::env::set_current_dir(&self.curdir).unwrap();
        let mut cmd = Command::new(&self.path);
        cmd.arg("--rfen").arg(rfen).arg("--ev1").arg(&self.evfile)
            .arg("--children").args(vec!["--depth".to_string(), format!("{depth}")])
            .args(&self.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if self.verbose {eprintln!("cmd:{cmd:?}");}
        cmd.spawn()
    }

    #[allow(dead_code)]
    pub fn run(&self, rfen : &str) -> Result<(String, String), String> {
        // launch another
        let curdir = std::env::current_dir().unwrap();
        let cmd = match self.spawn(rfen) {
            Err(msg) => {
                std::env::set_current_dir(curdir).unwrap();
                return Err(format!(
                    "error running ruversi... [{msg}], config:[{self}]"))
            },
            Ok(prcs) => prcs,
        };
        // read stdout and get moves
        let w = cmd.wait_with_output().unwrap();
        std::env::set_current_dir(curdir).unwrap();
        let txt = String::from_utf8(w.stdout).unwrap();
        // println!("txt:{txt}");
        let lines : Vec<_> = txt.split("\n").collect();
        if lines.len() < 13 {
            return Err(format!("invalid input {lines:?}"));
        }

        let res = lines[12].to_ascii_lowercase();
        if self.verbose {println!("opp:{}", &res);}
        let posptn = regex::Regex::new("nodes\\. ([A-Ha-h][1-8])").unwrap();
        let xtxt = match posptn.captures(&res) {
            Some(cap) => {
                String::from(&cap[1].to_lowercase())
            },
            _ => {
                return Err(
                    format!("invalid input from ruversi. \"{}\"", &res));
            }
        };

        let scoreptn = regex::Regex::new("val:(-?\\d+\\.\\d+) ").unwrap();
        match scoreptn.captures(&res) {
            Some(cap) => {
                Ok((xtxt, String::from(&cap[1])))
            },
            _ => {
                Err(format!("invalid input from ruversi. \"{}\" pos{xtxt}",
                    lines[2]))
            }
        }
    }

    /// `--rfen`と`--children`をつけて実行。
    /// 標準出力を処理して子供の局面の情報を返す。
    /// 
    /// # Arguments
    /// - rfen 開始局面。この局面の子供の局面の情報が返る。
    /// 
    /// # Returns
    /// 指定した局面の子供の局面の情報(Bitboardと最終結果)が返る。
    /// ダブり解消の処理のために確定石の欄にゼロを入れている。
    pub fn run_children(&self, rfen : &str)
            -> Result<Vec<(bitboard::BitBoard, i8, i8, i8)>, String> {
        let depth = bitboard::count_empty_cells(&rfen)?;
        // launch ruversi
        let curdir = std::env::current_dir().unwrap();
        let cmd = match self.spawn_children(rfen, depth as u32) {
            Err(msg) => {
                std::env::set_current_dir(curdir).unwrap();
                return Err(format!(
                    "error running ruversi... [{msg}], config:[{self}]"))
            },
            Ok(prcs) => prcs,
        };
        // read stdout and get moves
        let w = cmd.wait_with_output().unwrap();
        std::env::set_current_dir(curdir).unwrap();
        let txt = String::from_utf8(w.stdout).unwrap();
        // eprintln!("txt:{txt}");
        let lines : Vec<_> = txt.split("\n").collect();
        if lines.len() < 10 {
            return Err(format!("invalid input {lines:?}"));
        }

        // "val,{val:.2},{newban},{node}";
        // ex.
        // |__|__|__|__|__|__|__|__|  <-- 不要
        // @@'s turn.  <-- 不要
        // val:-0.3012 2185 nodes. D3c5D6e3F4c6F5 7msec  <-- 不要
        // val,-1.62,8/8/3A4/3B3/3Aa3/8/8/8 w,1769 nodes. c3C4c5B4d2C2a3
        // val,-1.62,8/8/8/2C3/3Aa3/8/8/8 w,507 nodes. c3D3
        // val,-1.50,8/8/8/3aA3/3C2/8/8/8 w,2357 nodes. d6C4f4C5f6G5d3
        // val,-1.13,8/8/8/3aA3/3B3/4A3/8/8 w,1262 nodes. f4D3e7F3e3F6d6
        let scoreptn = regex::Regex::new("val,([-0-9.]+),([0-8A-Ha-h\\/]+ [bw]),").unwrap();
        let ret = lines.iter().filter_map(|line| {
            match scoreptn.captures(&line) {
                Some(cap) => {
                    // cap[1] : val, cap[2] : rfen
                    Some((
                        bitboard::BitBoard::from(&cap[2]).ok()?,
                        0, 0,
                        cap[1].parse::<f32>().ok()? as i8
                    ))
                },
                _ => {None}
            }
        }).collect::<Vec<_>>();

        Ok(ret)
    }
}

#[test]
fn test_ruversirunner_default_values() {
    // RuversiRunner::new() で各フィールドがデフォルト値になっていることを確認
    let rr = RuversiRunner::new();
    assert_eq!(rr.curdir, "../ruversi");
    assert_eq!(rr.path, "./target/release/ruversi");
    assert_eq!(rr.evfile, "data/evaltable.txt");
    assert!(!rr.verbose);
}

#[test]
fn test_ruversirunner_set_verbose() {
    // set_verbose で verbose フィールドが変更されることを確認
    let mut rr = RuversiRunner::new();
    rr.set_verbose(false);
    assert!(!rr.verbose);
    rr.set_verbose(true);
    assert!(rr.verbose);
}

#[test]
fn test_ruversirunner_to_str() {
    // to_str の内容がフィールドに基づくことを確認
    let rr = RuversiRunner::new();
    let s = rr.to_string();
    assert!(s.contains("curdir:../ruversi"));
    assert!(s.contains("ruversi:./target/release/ruversi"));
    assert!(s.contains("evfile:data/evaltable.txt"));
}

#[test]
fn test_ruversirunner_read_config_file() {
    // 設定ファイルを作成し、read で値が読み込まれることを確認
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner_config.txt");
    let contents = "\
curdir:/tmp/myruversi
path:/tmp/myruversi_bin
evfile:/tmp/myruversi_evfile.txt
";
    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let mut rr = RuversiRunner::new();
    let result = rr.read(&config_path);
    assert!(result.is_ok());
    assert_eq!(rr.curdir, "/tmp/myruversi");
    assert_eq!(rr.path, "/tmp/myruversi_bin");
    assert_eq!(rr.evfile, "/tmp/myruversi_evfile.txt");
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn test_ruversirunner_read_config_file_empty_arg1() {
    // 一時ファイルに設定を書き込み、
    // read で値が読み込まれ、
    // 空文字の引数が含まれているErrが返る事を確認
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner_config_arg1.txt");
    let contents = "\
curdir:/tmp/myruversi
path:/tmp/myruversi_bin
evfile:/tmp/myruversi_evfile.txt
args:a,b,c,
";

    // 前のテストのファイルが残ってたら消す
    if config_path.exists() {
        println!("removed config file for test.");
        std::fs::remove_file(&config_path).unwrap();
    }

    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let mut rr = RuversiRunner::new();
    let result = rr.read(&config_path);
    assert_eq!(result, Err("\"a,b,c,\" contains empty part @3! [\"a\", \"b\", \"c\", \"\"]".to_string()));
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn test_ruversirunner_read_config_file_empty_arg2() {
    // 一時ファイルに設定を書き込み、
    // read で値が読み込まれ、
    // 空文字の引数が含まれているErrが返る事を確認
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner_config_arg2.txt");
    let contents = "\
curdir:/tmp/myruversi
path:/tmp/myruversi_bin
evfile:/tmp/myruversi_evfile.txt
args:a,b,,c
";

    // 前のテストのファイルが残ってたら消す
    if config_path.exists() {
        println!("removed config file for test.");
        std::fs::remove_file(&config_path).unwrap();
    }

    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let mut er = RuversiRunner::new();
    let result = er.read(&config_path);
    assert_eq!(result, Err("\"a,b,,c\" contains empty part @2! [\"a\", \"b\", \"\", \"c\"]".to_string()));
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn test_ruversirunner_read_config_file_empty_arg3() {
    // 一時ファイルに設定を書き込み、
    // read で値が読み込まれることを確認
    // args:の後ろが空白だけでもエラーに鳴らないことの確認
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner_config_arg3.txt");
    let contents = "\
curdir:/tmp/myruversi
path:/tmp/myruversi_bin
evfile:/tmp/myruversi_evfile.txt
args:  
";

    // 前のテストのファイルが残ってたら消す
    if config_path.exists() {
        println!("removed config file for test.");
        std::fs::remove_file(&config_path).unwrap();
    }

    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let mut er = RuversiRunner::new();
    let result = er.read(&config_path);
    assert_eq!(result, Ok(()));
    assert_eq!(er.curdir, "/tmp/myruversi");
    assert_eq!(er.path, "/tmp/myruversi_bin");
    assert_eq!(er.evfile, "/tmp/myruversi_evfile.txt");
    assert!(er.args.is_empty());
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn test_ruversirunner_read_config_file_empty_arg4() {
    // 一時ファイルに設定を書き込み、
    // read で値が読み込まれることを確認
    // args:の後ろが空でもエラーに鳴らないことの確認
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner_config_arg4.txt");
    let contents = "\
curdir:/tmp/myruversi
path:/tmp/myruversi_bin
evfile:/tmp/myruversi_evfile.txt
args:
";

    // 前のテストのファイルが残ってたら消す
    if config_path.exists() {
        println!("removed config file for test.");
        std::fs::remove_file(&config_path).unwrap();
    }

    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let mut er = RuversiRunner::new();
    let result = er.read(&config_path);
    assert_eq!(result, Ok(()));
    assert_eq!(er.curdir, "/tmp/myruversi");
    assert_eq!(er.path, "/tmp/myruversi_bin");
    assert_eq!(er.evfile, "/tmp/myruversi_evfile.txt");
    assert!(er.args.is_empty());
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn test_ruversirunner_read_config_file_not_found() {
    // 存在しないファイルを指定した場合、Errが返ることを確認
    let mut rr = RuversiRunner::new();
    let result = rr.read(
        &std::path::PathBuf::from("/tmp/no_such_ruversi_config.txt"));
    assert!(result.is_err());
}

#[test]
fn test_ruversirunner_from_config_empty() {
    // from_config("") でデフォルト値
    let rr = RuversiRunner::from_config(
        &std::path::PathBuf::from("")).unwrap();
    assert_eq!(rr.curdir, "../ruversi");
}

#[test]
fn test_ruversirunner_from_config_file() {
    // from_config で設定ファイルを読み取る
    let tmp = std::env::temp_dir();
    let config_path =
            tmp.join("test_ruversirunner2_config.txt");
    let contents = "curdir:/tmp/abc\n";
    {
        let cp = config_path.clone();
        let mut file = std::fs::File::create(cp).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }
    let rr = RuversiRunner::from_config(&config_path).unwrap();
    assert_eq!(rr.curdir, "/tmp/abc");
    std::fs::remove_file(config_path).unwrap();
}
