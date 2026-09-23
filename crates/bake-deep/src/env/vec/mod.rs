//! Vectorized environment support module
//! 
use crate::data::Batchable;
use burn::prelude::*;
/// trait for Vectorized Environment
pub trait VectorizedEnvironment {
    type Obs: Batchable;
    type Action: Batchable;
    type Constraint: Batchable;
    /// the number of environments
    fn n_envs(&self) -> usize;
    /// take a step from given actions
    fn step(&mut self, actions: Self::Action) -> ((Self::Obs, Self::Constraint), Tensor<1>, Tensor<1>, Tensor<1>);
    /// reset the terminated or truncated environment, which is determined by given indices
    fn reset(&mut self, indices: Tensor<1>) -> (Self::Obs, Self::Constraint);
    /// return the device
    fn device(&self) -> Device;
}

pub mod sync_env;
pub use sync_env::SynchronizedEnvironment;