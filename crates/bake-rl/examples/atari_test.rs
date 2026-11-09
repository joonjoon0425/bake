use bake_rl::core::env::space::Space;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::core::env::{Environment, EnvironmentConfig};
use rand::prelude::*;

pub fn main() {
    let mut env = GymnasiumEnvironmentConfig::mario_bros_preprocessed().kwarg("render_mode", "human").init();
    let mut rng = SmallRng::seed_from_u64(12);

    env.reset();
    for _ in 0..1000 {
        let action = env.action_space().sample(&mut rng);
        let ((_, _), _, ter, tru, _) = env.step(action);
        
        if ter || tru {
            env.reset();
        }
    }
}