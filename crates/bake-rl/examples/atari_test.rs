use bake_rl::core::env::space::{BatchedSpace, Space};
use bake_rl::core::env::{Environment, EnvironmentConfig, BatchedEnvironment, BatchedEnvironmentConfig};
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::env::gym;
use rand::prelude::*;

pub fn main() {
    let n_envs = 4;
    let device = burn::prelude::Device::flex();
    let mut env = gym::box2d::batched_lunar_lander_config().kwarg("render_mode", "human").init(n_envs, &device);
    let mut rng = SmallRng::seed_from_u64(12);

    env.reset(None);
    for _ in 0..1000 {
        let action = env.action_space().sample();
        let ((_, _), _, ter, tru, _) = env.step(action);
        
        env.reset(ter.bool_or(tru).into());
    }
}