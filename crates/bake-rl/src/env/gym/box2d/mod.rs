//! Box2D environments

use crate::{core::env::space::{ContinuousSpace, DiscreteSpace}, env::gym::GymnasiumEnvironmentConfig};

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// LunarLander-v3 environment
    pub fn lunarlander() -> Self {
        Self::new("LunarLander-v3")
    }
}

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// BipedalWalker-v3 environment
    pub fn bipedalwalker() -> Self {
        Self::new("BipedalWalker-v3")
    }
}