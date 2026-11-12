use bake_rl::core::env::space::BatchedSpace;
use bake_rl::env::wrapper::SynchronizedBatchedEnvironment;
use bake_rl::core::env::{EnvironmentConfig, BatchedEnvironment};
use bake_rl::env::gym::GymnasiumEnvironmentConfig;

pub fn main() {
    let n_envs = 6;
    let device = burn::prelude::Device::flex();
    let mut envs = vec![];
    for i in 0..n_envs { envs.push(GymnasiumEnvironmentConfig::pong_preprocessed().seed(i).init()) }
    let mut env = SynchronizedBatchedEnvironment::new(envs, &device);

    println!("obs shape: {:?}", env.obs_space().shape());
    println!("action numbers: {:?}", env.action_space().n());

    env.reset(None);
    for _ in 0..1000 {
        let action = env.action_space().sample();
        let ((_, _), _, ter, tru, _) = env.step(action);
        env.reset(ter.bool_or(tru).into());
    }
}