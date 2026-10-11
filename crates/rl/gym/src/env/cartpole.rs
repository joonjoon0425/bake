//! CartPole-v1
//! 

use bake_rl_env::space::{ContinuousSpace, DiscreteSpace};
use crate::env::GymnasiumEnvironmentConfig;

/// CartPole-v1
pub fn cartpole() -> GymnasiumEnvironmentConfig<ContinuousSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("CartPole-v1")
}