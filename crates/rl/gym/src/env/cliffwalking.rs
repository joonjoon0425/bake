//! CliffWalking-v1
//! 

use bake_rl_env::space::DiscreteSpace;
use crate::env::GymnasiumEnvironmentConfig;

/// CliffWalking-v1
pub fn cliff_walking() -> GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new("CliffWalking-v1")
}