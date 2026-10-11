//! MountainCar-v0
//! 

use bake_rl_env::space::{ContinuousSpace, DiscreteSpace};
use crate::env::GymnasiumEnvironmentConfig;

/// MountainCar-v0
pub fn mountain_car() -> GymnasiumEnvironmentConfig<ContinuousSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("MountainCar-v0")
}