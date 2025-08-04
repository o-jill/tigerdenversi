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

pub fn loadkifu(files : &[String], d : &str, progress : usize)
        -> Vec<(bitboard::BitBoard, i8, i8, i8, i8)> {
    // let sta = std::time::Instant::now();
    let boards = files.par_iter().flat_map(|fname| {
        let path = format!("{d}/{fname}");
        print!("{path}\r");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        kifu.list.par_iter().filter_map(|t| {
            let ban = bitboard::BitBoard::from(&t.rfen).unwrap();
            if ban.is_full() || !ban.is_progress(progress) {return None;}

            let (fsb, fsw) = ban.fixedstones();
            let score = kifu.score.unwrap();
            Some(vec![
                (ban.clone(), t.teban, fsb, fsw, score),
                // オーグメンテーション
                (ban.rotate90(), t.teban, fsb, fsw, score),
                (ban.rotate180(), t.teban, fsb, fsw, score),
                (ban.rotate180().rotate90(), t.teban, fsb, fsw, score),
                (ban.flip_horz(), t.teban, fsb, fsw, score),
                (ban.flip_vert(), t.teban, fsb, fsw, score),
                // flip color
                (ban.flip_all(), -t.teban, fsw, fsb, -score),
                (ban.rotate90().flip_all(), -t.teban, fsw, fsb, -score),
                (ban.rotate180().flip_all(), -t.teban, fsw, fsb, -score),
                (ban.rotate180().rotate90().flip_all(), -t.teban, fsw, fsb, -score),
                (ban.flip_horz().flip_all(), -t.teban, fsw, fsb, -score),
                (ban.flip_vert().flip_all(), -t.teban, fsw, fsb, -score)
            ])
        }).flatten().collect::<Vec<_>>()
    }).collect();
    println!();
    // println!("{}usec",sta.elapsed().as_micros());
    boards
}

pub fn dedupboards(boards : &mut Vec<(bitboard::BitBoard, i8, i8, i8, i8)>) {
    // println!("board: {} boards", boards.len());
    // let sta = std::time::Instant::now();
    boards.sort_by(|a, b| {
        a.0.black.cmp(&b.0.black).then(a.0.white.cmp(&b.0.white))
    });
    boards.dedup_by(|a, b| {a == b});
    // println!("{}usec",sta.elapsed().as_micros());
    println!("board: {} boards", boards.len());
}

pub fn extractboards(boards : &[(bitboard::BitBoard, i8, i8, i8, i8)])
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

pub fn extractscore(boards : &[(bitboard::BitBoard, i8, i8, i8, i8)]) -> Vec<f32> {
    boards.iter().map(|(_b, _t, _fb, _fw, s)| *s as f32).collect::<Vec<f32>>()
}
