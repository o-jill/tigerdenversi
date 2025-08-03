use super::*;

const INPUTSIZE :i64 = weight::N_INPUT as i64;

// list up kifu
pub fn findfiles(kifupath : &str) -> Vec<String> {
    let dir = std::fs::read_dir(kifupath).unwrap();
    let mut files = dir.filter_map(|entry| {
        entry.ok().and_then(|e|
            e.path().file_name().map(|n|
                n.to_str().unwrap().to_string()
            )
        )}).collect::<Vec<String>>().iter().filter(|&fnm| {
            fnm.contains("kifu")
            // fnm.contains(".txt")
        }).cloned().collect::<Vec<String>>();
    // println!("{:?}", files);

    files.sort();
    files
}

pub fn loadkifu(files : &[String], d : &str, progress : usize)
        -> Vec<(bitboard::BitBoard, i8, i8, i8, i8)> {
    let mut boards : Vec<(bitboard::BitBoard, i8, i8, i8, i8)> = Vec::new();
    for fname in files.iter() {
        let path = format!("{d}/{fname}");
        print!("{path}\r");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines:Vec<&str> = content.split('\n').collect();
        let kifu = kifu::Kifu::from(&lines);
        for t in kifu.list.iter() {
            let ban = bitboard::BitBoard::from(&t.rfen).unwrap();
            if ban.is_full() {continue;}
            if !ban.is_progress(progress) {continue;}

            let (fsb, fsw) = ban.fixedstones();

            let ban90 = ban.rotate90();
            let ban180 = ban.rotate180();
            let ban270 = ban180.rotate90();
            let banh = ban.flip_horz();
            let banv = ban.flip_vert();
            let banfa = ban.flip_all();
            let ban90fa = ban90.flip_all();
            let ban180fa = ban180.flip_all();
            let ban270fa = ban270.flip_all();
            let banhfa = banh.flip_all();
            let banvfa = banv.flip_all();
            let score = kifu.score.unwrap();
            boards.push((ban, t.teban, fsb, fsw, score));
            boards.push((ban90, t.teban, fsb, fsw, score));
            boards.push((ban180, t.teban, fsb, fsw, score));
            boards.push((ban270, t.teban, fsb, fsw, score));
            boards.push((banh, t.teban, fsb, fsw, score));
            boards.push((banv, t.teban, fsb, fsw, score));
            /* flip color */
            boards.push((banfa, -t.teban, fsw, fsb, -score));
            boards.push((ban90fa, -t.teban, fsw, fsb, -score));
            boards.push((ban180fa, -t.teban, fsw, fsb, -score));
            boards.push((ban270fa, -t.teban, fsw, fsb, -score));
            boards.push((banhfa, -t.teban, fsw, fsb, -score));
            boards.push((banvfa, -t.teban, fsw, fsb, -score));
        }
    }
    println!();
    boards
}

pub fn dedupboards(boards : &mut Vec<(bitboard::BitBoard, i8, i8, i8, i8)>) {
    boards.sort_by(|a, b| {
        a.0.black.cmp(&b.0.black).then(a.0.white.cmp(&b.0.white))
    });
    boards.dedup_by(|a, b| {a == b});
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
