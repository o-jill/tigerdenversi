use std::io::Write;
use clap::Parser;
use rand::seq::SliceRandom;
use tch::nn::{self, Module, VarStore};
use tch::Tensor;

mod kifu;
mod bitboard;
mod weight;
mod argument;
mod data_loader;
mod neuralnet;
mod training;
mod ruversirunner;

fn main() -> Result<(), tch::TchError> {
    let arg = argument::Arg::parse();
    let extract_mate = arg.extract_mate_n();

    let mut train = training::Training::from(arg);

    if extract_mate != 0 {
        return train.extract_mate(extract_mate);
    }

    train.run()?;

    train.write();

    Ok(())
}
