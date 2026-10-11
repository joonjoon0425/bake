//! LunarLander-v3
//! 

use bake_rl_env::space::{ContinuousSpace, DiscreteSpace};
use crate::env::GymnasiumEnvironmentConfig;

/// LunarLander-v3
pub fn lunar_lander() -> GymnasiumEnvironmentConfig<ContinuousSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("LunarLander-v3")
}