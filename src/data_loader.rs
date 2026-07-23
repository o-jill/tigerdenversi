use super::*;

use rand::Rng;
use rayon::prelude::*;

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const PROGRESS_ALLOW_ALL : usize = usize::MAX;

// clean up training data in path
pub fn clean_up_large_data(path : &str) -> Result<(), String> {
    let dirpath = std::path::Path::new(path);
    if !dirpath.exists() {return Ok(());}

    let dir = std::fs::read_dir(dirpath).unwrap();
    for entry in dir {
        if let Ok(e) = &entry {
            if let Ok(m) = e.metadata() {
                if !m.is_file() {
                    continue;
                }

                if let Err(emsg) = std::fs::remove_file(e.path()) {
                    return Err(format!("remove_file:{:?} error:{emsg}", e.path()));
                }
            }
        }
    }

    Ok(())
}

pub fn gen_div_file_name(d : &std::path::Path, n : usize) -> std::path::PathBuf {
    let fname = format!("kifu_part_{n:03}.txt");
    d.join(fname)
}

/// read data and store into files.
///
/// # Arguments
/// - `input_dir`
///   a directory which contains kifu*.txt.
/// - `output_dir`
///   a directory which will have kifu_part_*.txt.
/// - `div_ratio`
///   number of division.
///
/// # Returns
/// - `Ok(())`
///   successfully done.
/// - `Err(String)`
///   some error occurred.
pub fn prepare_large_data(input_dir : &str, output_dir : &str, div_ratio : i32)
        -> Result<(), String> {
    if input_dir.is_empty() || output_dir.is_empty() {return Ok(());}
    if div_ratio <= 0 {
        return Err("invalid ratio value: {div_ratio:?}".to_string());
    }

    // create output directory if it does not exist
    let dirpath = std::path::Path::new(output_dir);
    if !dirpath.exists() {
        if let Err(e) = std::fs::create_dir(dirpath) {
            return Err(
                format!("failed to create directory: \"{output_dir}\" error:{e}"));
        }
    }

    let (tx, rx) =
        std::sync::mpsc::channel::<Vec<(bitboard::BitBoard, i8)>>();
    let outdir = output_dir.to_string();
    let div_size = div_ratio as usize;
    // shuffle and file store
    let th = std::thread::spawn(move|| {
        let dirpath = std::path::Path::new(&outdir);
        let mut rng = rand::thread_rng();
        let mut buffers: Vec<String> = vec![String::new() ; div_size];
        loop {
            let data = rx.recv().unwrap();
            if data.is_empty() {break;}

            for (ban, score) in data {
                // content: rfen,score\n
                let text = format!("{},{score}\n", ban.to_string_short());
                let n = rng.gen_range(0..div_size);
                buffers[n] += &text;
                const LIMIT_SIZE : usize = 1024 * 50;
                if buffers[n].len() < LIMIT_SIZE {continue;}

                let filepath = gen_div_file_name(dirpath, n);
                let mut f = std::fs::OpenOptions::new()
                    .create(true).append(true).open(filepath).unwrap();
                f.write_all(buffers[n].as_bytes()).unwrap();
                buffers[n].clear();
            }
        }

        // flush remaining data
        for (n, buf) in buffers.iter().enumerate() {
            let filepath = gen_div_file_name(dirpath, n);
            let mut f = std::fs::OpenOptions::new()
                .create(true).append(true).open(filepath).unwrap();
            f.write_all(buf.as_bytes()).unwrap();
        }
    });

    let files = find_kifu_files(input_dir);
    for fname in files {
        let path = std::path::Path::new(input_dir).join(fname);
        // eprintln!("path:{path:?}");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        for te in kifu.list {
            let ban = bitboard::BitBoard::try_from(te.rfen.as_str()).unwrap();
            // 最後の局面とか後一手の局面とかは不要
            if ban.is_last1_or_full() {continue;}

            let score = kifu.score.unwrap();
            let ret = ban.rotated_mirrored(score);
            if ret.is_empty() {continue;}

            // ファイルに出力する
            tx.send(ret).unwrap();
        }
    }
    tx.send(Vec::new()).unwrap();
    th.join().unwrap();

    Ok(())
}

/// read data and store into files.
///
/// # Arguments
/// - `input_dir`
///   a directory which contains mate files.
/// - `input_files`
///   mate files
/// - `output_dir`
///   a directory which will have kifu_part_*.txt.
/// - `div_ratio`
///   number of division.
///
/// # Returns
/// - `Ok(())`
///   successfully done.
/// - `Err(String)`
///   some error occurred.
pub fn prepare_large_mate(input_dir : &[String], input_files : &[String], output_dir : &str, div_ratio : i32)
        -> Result<(), String> {
    if output_dir.is_empty() {return Ok(());}
    if div_ratio <= 0 {
        return Err("invalid ratio value: {div_ratio:?}".to_string());
    }

    // create output directory if it does not exist
    let dirpath = std::path::Path::new(output_dir);
    if !dirpath.exists() {
        if let Err(e) = std::fs::create_dir(dirpath) {
            return Err(
                format!("failed to create directory: \"{output_dir}\" error:{e}"));
        }
    }

    let (tx, rx) =
        std::sync::mpsc::channel::<Vec<(bitboard::BitBoard, i8)>>();
    let outdir = output_dir.to_string();
    let div_size = div_ratio as usize;
    // shuffle and file store
    let th = std::thread::spawn(move|| {
        let dirpath = std::path::Path::new(&outdir);
        let mut rng = rand::thread_rng();
        let mut buffers: Vec<String> = vec![String::new() ; div_size];
        loop {
            let data = rx.recv().unwrap();
            if data.is_empty() {break;}

            for (ban, score) in data {
                // content: rfen,score\n
                let text = format!("{},{score}\n", ban.to_string_short());
                let n = rng.gen_range(0..div_size);
                buffers[n] += &text;
                const LIMIT_SIZE : usize = 1024 * 50;
                if buffers[n].len() < LIMIT_SIZE {continue;}

                let filepath = gen_div_file_name(dirpath, n);
                let mut f = std::fs::OpenOptions::new()
                    .create(true).append(true).open(filepath).unwrap();
                f.write_all(buffers[n].as_bytes()).unwrap();
                buffers[n].clear();
            }
        }

        // flush remaining data
        for (n, buf) in buffers.iter().enumerate() {
            let filepath = gen_div_file_name(dirpath, n);
            let mut f = std::fs::OpenOptions::new()
                .create(true).append(true).open(filepath).unwrap();
            f.write_all(buf.as_bytes()).unwrap();
        }
    });

    for inputdir in input_dir {
        let files = find_mate_files(inputdir);
        for fname in files {
            let path = std::path::Path::new(inputdir).join(fname);
            let content =
                load_mates_all(path.to_str().unwrap())
                    .map_err(|e| format!("err:{e}"))?;

            // ファイルに出力する
            tx.send(content).unwrap();
        }
    }

    for fname in input_files {
        let path = std::env::current_dir().unwrap().join(fname);
        let content =
            load_mates_all(path.to_str().unwrap())
                .map_err(|e| format!("err:{e}"))?;

        // ファイルに出力する
        tx.send(content).unwrap();
    }

    tx.send(Vec::new()).unwrap();
    th.join().unwrap();

    Ok(())
}

/// list up files
///
/// # Arguments
/// - `path` directory path to find files.
/// - `pattern` find files which contains `pattern` is their name.
///
/// # Returns
/// Vec of names of files.
pub fn findfiles(path : &str, pattern : &str) -> Vec<String> {
    // let sta = std::time::Instant::now();
    let dir = std::fs::read_dir(path).unwrap();
    let mut files = dir.filter_map(|entry| {
        if let Ok(e) = &entry {
            if let Ok(m) = e.metadata() {
                if !m.is_file() {
                    return None;
                }
            }
        }
        entry.ok().and_then(|e|
            e.path().file_name().map(|n|
                n.to_str().unwrap().to_string()
            )
        )}).filter(|fnm| {
            fnm.contains(pattern)
            // fnm.contains(".txt")
        }).collect::<Vec<String>>();
    // println!("{:?}", files);

    files.sort();
    // println!("{}usec",sta.elapsed().as_micros());
    files
}


/// list up kifu
///
/// # Arguments
/// - `matepath` directory path to find kifu files.
///
/// # Returns
/// Vec of names of files.
pub fn find_kifu_files(kifupath : &str) -> Vec<String> {
    findfiles(kifupath, "kifu")
}

/// list up mate
///
/// # Arguments
/// - `matepath` directory path to find mate files.
///
/// # Returns
/// Vec of names of files.
pub fn find_mate_files(matepath : &str) -> Vec<String> {
    findfiles(matepath, "mate")
}

pub fn loadkifu(files : &[String], d : &str, progress : usize,
        log : &mut std::fs::File, show_path : bool)
            -> Vec<(bitboard::BitBoard, i8)> {
    // let sta = std::time::Instant::now();
    let shared = std::sync::Mutex::new(log);
    let boards = files.par_iter().flat_map(|fname| {
        let path = std::path::Path::new(d).join(fname);
        {
            let mut l = shared.lock().unwrap();
            l.write_all(format!("{}\n", path.display()).as_bytes()).unwrap();
            if show_path {print!("{}\r", path.display());}
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        kifu.list.par_iter().filter_map(|t| {
            let ban = bitboard::BitBoard::try_from(t.rfen.as_str()).unwrap();
            // 最後の局面とか後一手の局面とか覚えたい進行度の時じゃない
            if ban.is_last1_or_full() || !ban.is_progress(progress) {
                return None;
            }

            let score = kifu.score.unwrap();
            let mut ret = ban.rotated_mirrored(score);

            if !cfg!(feature = "extract_mate3") {return Some(ret);}

            // 残り3つだったら全部の着手パターンも生成して登録する。
            if !ban.is_last_n(3) {return Some(ret);}

            let mvs = ban.genmove().unwrap();
            if mvs.len() <= 1 {return Some(ret);}

            for mvxy in mvs {
                // newbanはmate2
                // この局面の評価を計算して登録する
                let newban = ban.r#move(mvxy).unwrap();
                let mvs2 = newban.genmove().unwrap();
                if mvs2.is_empty() || mvs2[0] == bitboard::PASS {  // pass
                    // panic!("mvs2.is_empty() stones:{}", newban.stones());
                    // skip solving.
                    continue;
                }

                let scores = mvs2.iter().map(|mvxy2| {
                    // newban2はmate1
                    let newban2 = newban.r#move(*mvxy2).unwrap();
                    if !newban2.is_last1() {panic!("!newban2.is_last1()");}

                    let (val, _) = newban2.move_mate1();
                    val as i8
                }).collect::<Vec<_>>();
                if scores.is_empty() {panic!("scores.is_empty()");}

                let score = *scores.iter().reduce(|a, b| a.max(b)).unwrap();
                let mut aug = newban.rotated_mirrored(score);
                ret.append(&mut aug);
            }
            Some(ret)
        }).flatten().collect::<Vec<_>>()
    }).collect();
    if show_path {println!();}
    // println!("{}usec",sta.elapsed().as_micros());
    boards
}

fn read_mate_file_augmentation(buf : impl std::io::BufRead, progress : usize)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let mut ret = Vec::new();

    for line in buf.lines() {
        match line {
            Err(e) => {return Err(format!("{e}"))},
            Ok(l) => {
                // コメント行 or 7文字未満
                if l.len() < 7 || l.starts_with("#") {continue;}

                // rfen,score
                let elem : Vec<&str> = l.split(",").collect();
                if elem.len() < 2 {
                    return Err(format!("# of elem < 2 w/ {l}!"));
                }

                let ban = match bitboard::BitBoard::try_from(elem[0]) {
                    Ok(b) => {b},
                    Err(msg) => {
                        return Err(format!("error: {msg} @ {}", elem[0]));
                    },
                };
                if progress != PROGRESS_ALLOW_ALL && !ban.is_progress(progress) {continue;}

                // let (b, w) = ban.fixedstones();
                let score = match elem[1].parse::<i8>() {
                    Err(msg) => {
                        return Err(format!("error: parse score : {msg}"));
                    },
                    Ok(num) => {num},
                };

                ret.append(&mut ban.rotated_mirrored(score));
            }
        }
    }

    Ok(ret)
}

fn read_mate_file(buf : impl std::io::BufRead, progress : usize)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let mut ret = Vec::new();

    for line in buf.lines() {
        match line {
            Err(e) => {return Err(format!("{e}"))},
            Ok(l) => {
                // コメント行 or 7文字未満
                if l.len() < 7 || l.starts_with("#") {continue;}

                // rfen,score
                let elem : Vec<&str> = l.split(",").collect();
                if elem.len() < 2 {
                    return Err(format!("# of elem < 2 w/ {l}!"));
                }

                let ban = match bitboard::BitBoard::try_from(elem[0]) {
                    Ok(b) => {b},
                    Err(msg) => {
                        return Err(format!("error: {msg} @ {}", elem[0]));
                    },
                };
                if progress != PROGRESS_ALLOW_ALL && !ban.is_progress(progress) {continue;}

                // let (b, w) = ban.fixedstones();
                let score = match elem[1].parse::<i8>() {
                    Err(msg) => {
                        return Err(format!("error: parse score : {msg}"));
                    },
                    Ok(num) => {num},
                };

                ret.push((ban, score));
            }
        }
    }

    Ok(ret)
}

pub fn load_mates_augmentation(path : &str, progress : usize)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let filepath = std::path::Path::new(path);
    if !filepath.exists() {return Err(format!("{path} does NOT exist!"));}

    if path.ends_with(".zst") || path.ends_with(".zstd") {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("error: {e} @ File::open"))?;
        let z = zstd::Decoder::new(f)
            .map_err(|e| format!("error: {e} @ zstd::Decoder::new"))?;

        let buf = std::io::BufReader::new(z);
        read_mate_file_augmentation(buf, progress)
    } else {
        let f = std::fs::File::open(path).map_err(|e| format!("{e}"))?;

        let buf = std::io::BufReader::new(f);
        read_mate_file_augmentation(buf, progress)
    }
}

pub fn load_mates(path : &str, progress : usize)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let filepath = std::path::Path::new(path);
    if !filepath.exists() {return Err(format!("{path} does NOT exist!"));}

    if path.ends_with(".zst") || path.ends_with(".zstd") {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("error: {e} @ File::open"))?;
        let z = zstd::Decoder::new(f)
            .map_err(|e| format!("error: {e} @ zstd::Decoder::new"))?;

        let buf = std::io::BufReader::new(z);
        read_mate_file(buf, progress)
    } else {
        let f = std::fs::File::open(path).map_err(|e| format!("{e}"))?;

        let buf = std::io::BufReader::new(f);
        read_mate_file(buf, progress)
    }
}

fn load_mates_all(path : &str)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let filepath = std::path::Path::new(path);
    if !filepath.exists() {return Err(format!("{path} does NOT exist!"));}

    if path.ends_with(".zst") || path.ends_with(".zstd") {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("error: {e} @ File::open"))?;
        let z = zstd::Decoder::new(f)
            .map_err(|e| format!("error: {e} @ zstd::Decoder::new"))?;

        let buf = std::io::BufReader::new(z);
        read_mate_file_augmentation(buf, PROGRESS_ALLOW_ALL)
    } else {
        let f = std::fs::File::open(path).map_err(|e| format!("{e}"))?;

        let buf = std::io::BufReader::new(f);
        read_mate_file_augmentation(buf, PROGRESS_ALLOW_ALL)
    }
}

pub fn dedup_boards(boards : &mut Vec<(bitboard::BitBoard, i8)>) {
    // println!("board: {} boards", boards.len());
    // let sta = std::time::Instant::now();
    boards.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap()
    });
    boards.dedup_by(|a, b| {a == b});
    // println!("{}usec",sta.elapsed().as_micros());
}

pub fn extractboards(boards : &[(bitboard::BitBoard, i8)])
        -> Vec<f32> {
    boards.iter().map(|(b, _s)| {
        let mut v = [0.0f32 ; INPUTSIZE as usize];
        for y in 0..8 {
            for x in 0..8 {
                v[x + bitboard::NUMCELL * y] = b.black_at(x, y);
                v[x + bitboard::NUMCELL * y + weight::N_INPUT_BLACK] = b.white_at(x, y);
            }
        }
        v
    }).collect::<Vec<[f32 ; INPUTSIZE as usize]>>().concat()
}

pub fn extractscore(boards : &[(bitboard::BitBoard, i8)]) -> Vec<f32> {
    boards.iter().map(|(_b, s)| *s as f32).collect::<Vec<f32>>()
}

#[test]
fn test_extract_boards() {
    let input = [
        ("8/8/8/3Aa3/3aA3/8/8/8 b", 10i8), ("h/h/h/h/H/H/H/H w", 3i8),
        ("Ag/Ga/Bf/Fb/Ce/Ec/Dd/dD b",-2i8)
    ].iter().map(|(rfen, result)| {
        let ban = bitboard::BitBoard::try_from(*rfen).unwrap();
        (ban, *result)
    }).collect::<Vec<(bitboard::BitBoard, i8)>>();
    let convert = extractboards(&input);
    let answer = vec![
        // 8/8/8/3Aa3/3aA3/8/8/8 b"
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,1f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,1f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,1f32,0f32,0f32,0f32,
        0f32,0f32,0f32,1f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        // "h/h/h/h/H/H/H/H w"
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        // "Ag/Ga/Bf/Fb/Ce/Ec/Dd/dD b"
        1f32,0f32,0f32,0f32,0f32,0f32,0f32,0f32,
        1f32,1f32,1f32,1f32,1f32,1f32,1f32,0f32,
        1f32,1f32,0f32,0f32,0f32,0f32,0f32,0f32,
        1f32,1f32,1f32,1f32,1f32,1f32,0f32,0f32,
        1f32,1f32,1f32,0f32,0f32,0f32,0f32,0f32,
        1f32,1f32,1f32,1f32,1f32,0f32,0f32,0f32,
        1f32,1f32,1f32,1f32,0f32,0f32,0f32,0f32,
        0f32,0f32,0f32,0f32,1f32,1f32,1f32,1f32,
        0f32,1f32,1f32,1f32,1f32,1f32,1f32,1f32,
        0f32,0f32,0f32,0f32,0f32,0f32,0f32,1f32,
        0f32,0f32,1f32,1f32,1f32,1f32,1f32,1f32,
        0f32,0f32,0f32,0f32,0f32,0f32,1f32,1f32,
        0f32,0f32,0f32,1f32,1f32,1f32,1f32,1f32,
        0f32,0f32,0f32,0f32,0f32,1f32,1f32,1f32,
        0f32,0f32,0f32,0f32,1f32,1f32,1f32,1f32,
        1f32,1f32,1f32,1f32,0f32,0f32,0f32,0f32,
    ];

    assert_eq!(convert, answer);

    let scores = extractscore(&input);
    let answer = vec![10f32, 3f32, -2f32];
    assert_eq!(scores, answer);
}
