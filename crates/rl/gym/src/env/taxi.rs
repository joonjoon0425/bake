//! Taxi-v4
//! 

use bake_rl_env::space::DiscreteSpace;
use crate::env::GymnasiumEnvironmentConfig;

/// Taxi-v4
pub fn taxi() -> GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("Taxi-v4")
}