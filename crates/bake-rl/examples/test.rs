use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::prelude::*;
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use burn::prelude::*;

pub fn main() {
    let device = Device::default();
    let env = SynchronizedVectorEnvironmentConfig::new(vec![
        GymnasiumEnvironmentConfig::bipedalwalker().kwarg("render_mode", "human").seed(10) ,
    ]).init(&device);

    let mut tape = Tape::new(env, true);

    for _ in 0..100 {
        let _ = tape.step(
            tape.action_space().sample(1)
        );
    }
}