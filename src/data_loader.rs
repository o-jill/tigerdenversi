use super::*;

use rayon::prelude::*;

const INPUTSIZE :i64 = weight::N_INPUT as i64;

// list up kifu
pub fn findfiles(kifupath : &str) -> Vec<String> {
    // let sta = std::time::Instant::now();
    let dir = std::fs::read_dir(kifupath).unwrap();
    let mut files = dir.filter_map(|entry| {
        entry.ok().and_then(|e|
            e.path().file_name().map(|n|
                n.to_str().unwrap().to_string()
            )
        )}).filter(|fnm| {
            fnm.contains("kifu")
            // fnm.contains(".txt")
        }).collect::<Vec<String>>();
    // println!("{:?}", files);

    files.sort();
    // println!("{}usec",sta.elapsed().as_micros());
    files
}

pub fn loadkifu(files : &[String], d : &str, progress : usize,
        log : &mut std::fs::File, show_path : bool)
            -> Vec<(bitboard::BitBoard, i8)> {
    // let sta = std::time::Instant::now();
    let shared = std::sync::Mutex::new(log);
    let boards = files.par_iter().flat_map(|fname| {
        let path = format!("{d}/{fname}");
        {
            let mut l = shared.lock().unwrap();
            l.write_all(format!("{path}\n").as_bytes()).unwrap();
            if show_path {print!("{path}\r");}
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

fn read_mate_file(buf : impl std::io::BufRead, progress : usize)
        -> Result<Vec<(bitboard::BitBoard, i8)>, String> {
    let mut ret = Vec::new();

    for line in buf.lines() {
        match line {
            Err(e) => {return Err(format!("{e}"))},
            Ok(l) => {
                // コメント行 or 11文字未満
                if l.len() < 11 || l.starts_with("#") {continue;}
                // rfen,score
                let elem : Vec<&str> = l.split(",").collect();
                let ban = bitboard::BitBoard::try_from(elem[0])?;
                if !ban.is_progress(progress) {continue;}

                // let (b, w) = ban.fixedstones();
                let score = match elem[1].parse::<i8>() {
                    Err(msg) => {
                        return Err(format!("error: parse score : {msg}"));
                    },
                    Ok(num) => {num},
                };
                const AUGMENTATION_READ_MATE : bool = true;
                if AUGMENTATION_READ_MATE {
                    ret.append(&mut ban.rotated_mirrored(score));
                } else {
                    ret.push((ban, score));
                }
            }
        }
    }

    Ok(ret)
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

pub fn dedupboards(boards : &mut Vec<(bitboard::BitBoard, i8)>,
                   log : &mut std::fs::File, show_path : bool) {
    // println!("board: {} boards", boards.len());
    // let sta = std::time::Instant::now();
    boards.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap()
    });
    boards.dedup_by(|a, b| {a == b});
    // println!("{}usec",sta.elapsed().as_micros());
    let msg = format!("board: {} boards\n", boards.len());
    log.write_all(msg.as_bytes()).unwrap();
    if show_path {print!("{msg}");}
}

#[cfg(feature = "fixed_stones")]
pub fn extractboards(boards : &[(bitboard::BitBoard, i8)])
        -> Vec<f32> {
    boards.iter().map(|(b, fb, fw, _s)| {
        let mut v = [0.0f32 ; INPUTSIZE as usize];
        for y in 0..8 {
            for x in 0..8 {
                v[x + bitboard::NUMCELL * y + weight::N_INPUT_BLACK] = b.black_at(x, y);
                v[x + bitboard::NUMCELL * y + weight::N_INPUT_WHITE] = b.white_at(x, y);
            }
        }
        v[weight::N_INPUT_TEBAN] = b.teban as f32;
        v[weight::N_INPUT_FB] = *fb as f32;
        v[weight::N_INPUT_FW] = *fw as f32;
        v
    }).collect::<Vec<[f32 ; INPUTSIZE as usize]>>().concat()
}

#[cfg(not(feature = "fixed_stones"))]
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
