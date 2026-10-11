use bake_rl_env::{Environment, EnvironmentConfiguration};
use bake_rl_env_gym::env::cartpole;

pub fn main() {
    let mut env = cartpole().kwarg("render_mode", ("human", )).init();

    for _ in 0..100 {
        env.step(1);
    }
}