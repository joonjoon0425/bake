//! toy text environments

use crate::{core::env::space::DiscreteSpace, env::gym::GymnasiumEnvironmentConfig};

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// Taxi-v4 environment
    pub fn taxi() -> Self {
        Self::new("Taxi-v4")
    }
    /// FrozenLake-v1 environment
    pub fn frozenlake() -> Self {
        Self::new("FrozenLake-v1")
    }
    /// CliffWalking-v1 environment
    pub fn cliffwalking() -> Self {
        Self::new("CliffWalking-v1")
    }
}