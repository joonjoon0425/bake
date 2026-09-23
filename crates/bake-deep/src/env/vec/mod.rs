//! Vectorized environment support module
//! 
use crate::data::Batchable;
use burn::prelude::*;
/// trait for Vectorized Environment
pub trait VectorizedEnvironment {
    /// the type of observation
    type Obs: Batchable;
    /// the type of action
    type Action: Batchable;
    /// the type of constraints
    type Constraint: Batchable;
    /// the number of environments
    fn n_envs(&self) -> usize;
    /// take a step from given actions
    /// # Warning
    /// - the environment which is done will automatically reset, and follows the same_step method of Gymnasium
    fn step(&mut self, actions: Self::Action) -> ((Self::Obs, Self::Constraint), Tensor<1>, Tensor<1>, Tensor<1>, (Self::Obs, Self::Constraint));
    /// reset all environments
    fn reset_all(&mut self) -> (Self::Obs, Self::Constraint);
    /// return the device
    fn device(&self) -> Device;
}

pub mod sync_env;
pub use sync_env::SynchronizedEnvironment;