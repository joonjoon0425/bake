use bake_rl::core::env::space::{ContinuousSpace, DiscreteSpace};
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::prelude::*;
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use bake_rl::env::classic::cartpole::CartPoleConfig;
use burn::prelude::*;

pub fn main() {
    let device = Device::default();
    let env = SynchronizedVectorEnvironmentConfig::new(vec![
        GymnasiumEnvironmentConfig::<ContinuousSpace<2>, DiscreteSpace>::new("CartPole-v1"),
        GymnasiumEnvironmentConfig::<ContinuousSpace<2>, DiscreteSpace>::new("CartPole-v1"),
        GymnasiumEnvironmentConfig::<ContinuousSpace<2>, DiscreteSpace>::new("CartPole-v1")
    ]).init(&device);

    let mut tape = Tape::new(env, true);

    for _ in 0..10 {
        let t = tape.step(
            tape.action_space().sample(3)
        );
        println!("{:?}", t);
    }
}