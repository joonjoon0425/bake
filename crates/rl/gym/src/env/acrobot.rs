//! Acrobot-v1
//! 

use bake_rl_env::space::{ContinuousSpace, DiscreteSpace};
use crate::env::GymnasiumEnvironmentConfig;

/// Acrobot-v1
pub fn acrobot() -> GymnasiumEnvironmentConfig<ContinuousSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("Acrobot-v1")
}