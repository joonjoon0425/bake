//! FrozenLake-v1
//! 

use bake_rl_env::space::DiscreteSpace;
use crate::env::GymnasiumEnvironmentConfig;

/// FrozenLake-v1
pub fn frozen_lake() -> GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("FrozenLake-v1")
}