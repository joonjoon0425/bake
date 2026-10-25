use bake_rl::{core::{data::BatchTensor, env::{Tape, VectorizedEnvironmentConfig, wrapper::sync_env::SynchronizedVectorEnvironmentConfig}}, env::classic::cartpole::CartPoleConfig};
use burn::prelude::*;

pub fn main() {
    let device = Device::default();
    let env = SynchronizedVectorEnvironmentConfig::new(vec![CartPoleConfig::new().seed(1), CartPoleConfig::new().seed(13), CartPoleConfig::new().seed(2)]).init(&device);

    let mut tape = Tape::new(env, true);

    for _ in 0..10 {
        let t = tape.step(BatchTensor::with_batch_dim(vec![3], Tensor::from_ints([0, 1, 0], &device)));
        println!("{:?}", t);
    }
}