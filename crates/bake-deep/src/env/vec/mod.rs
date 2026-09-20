//! Vectorized environment support module
//! 
use crate::env::Environment;
use burn::prelude::*;
/// trait for Vectorized Environment
pub trait VectorizedEnvironment<E: Environment> {
    /// the number of environments
    fn n_envs(&self) -> usize;
    /// take a step from given actions
    fn step(&mut self, actions: E::Action) -> Vec<((E::Obs, E::Constraint), f32, bool, bool)>;
    /// reset the environment of given index
    fn reset(&mut self, index: usize) -> (E::Obs, E::Constraint);
    /// return the device
    fn device(&self) -> Device;
}

pub mod sync_env;
pub use sync_env::SynchronizedEnvironment;