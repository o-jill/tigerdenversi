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


fn main() -> Result<(), tch::TchError> {
    let arg = argument::Arg::parse();
    let extract_mate3 = arg.extract_mate3;
    let mut train = training::Training::from(arg);
    if extract_mate3 {
        return train.extract_mate3();
    }

    train.run()?;

    train.write();

    Ok(())
}
