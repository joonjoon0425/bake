use bake_rl::{core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig, env::gym::GymnasiumEnvironmentConfig, prelude::*};
use bake_rl::burn::prelude::*;

pub fn main() {
    let device = Device::default();
    let env = SynchronizedVectorEnvironmentConfig::new(
        vec![GymnasiumEnvironmentConfig::walker2d().seed(140).kwarg("render_mode", "human")]
    ).init(&device);

    let mut tape = Tape::new(env, true);

    for _ in 0..1000 {
        tape.step(
            tape.action_space().sample(1)
        );
    }
}