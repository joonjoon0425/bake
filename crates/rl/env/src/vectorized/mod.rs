//! A collection and basic trait for vectorized environments
//! 
use crate::space::Space;
use bake_rl_core::data::deep::Batchable;
use burn::prelude::*;
/// trait for Vectorized Environment
pub trait VectorizedEnvironment {
    /// the type of observation
    type ObsSpace: Space<Element: Batchable>;
    /// the type of action
    type ActionSpace: Space<Element: Batchable>;
    /// the type of constraints
    type Constraint: Batchable;
    /// the number of environments
    fn n_envs(&self) -> usize;
    /// take a step from given actions
    /// # Warning
    /// - the environment which is done will automatically reset, and follows the same_step method of Gymnasium
    fn step(&mut self, actions: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Tensor<1>, Tensor<1>, Tensor<1>, (<Self::ObsSpace as Space>::Element, Self::Constraint));
    /// reset all environments
    fn reset_all(&mut self) -> (<Self::ObsSpace as Space>::Element, Self::Constraint);
    /// return the device
    fn device(&self) -> Device;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;
}

pub mod tape;
pub use tape::Tape;

pub mod wrapper;
pub use wrapper::*;