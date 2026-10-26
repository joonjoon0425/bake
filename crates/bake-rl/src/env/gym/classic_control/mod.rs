//! Classic control environments

use crate::{core::env::space::{ContinuousSpace, DiscreteSpace}, env::gym::GymnasiumEnvironmentConfig};

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// CartPole-v1 environment
    pub fn cartpole() -> Self {
        Self::new("CartPole-v1")
    }
    /// MountainCar-v0 environment
    pub fn mountaincar() -> Self {
        Self::new("MountainCar-v0")
    }
    /// Acrobot-v1 environment
    pub fn acrobot() -> Self {
        Self::new("Acrobot-v1")
    }
}

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// MountainCarContinuous-v0 environment
    pub fn mountaincar_continuous() -> Self {
        Self::new("MountainCarContinuous-v0")
    }
    /// Pendulum-v1 environment
    pub fn pendulum() -> Self {
        Self::new("Pendulum-v1")
    }
}