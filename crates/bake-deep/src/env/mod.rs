//! A native rust environments for deep rl
//! 

use burn::tensor::Device;

/// ### A trait which all environments must implement
/// ##### Warning
/// All environments should return the obs with batch dimension, and receive action with batch dimension.
/// The first principle of this framework is that everything owns a batch dimension.
/// This is due to the fixed rank of burn's `Tensor`.
/// Also, environments must produce the non-autodiff tensors.
pub trait Environment {
    /// The observation type which environment produces
    type Obs: Batchable;
    /// The action type which environment receives
    type Action: Batchable;
    /// The constraint for actions. If the environment does not provides constraint, set it as `Unconstrained`
    type Constraint: Batchable;

    /// reset the environment
    fn reset(&mut self) -> (Self::Obs, Self::Constraint);
    /// take one step and return a tuple of ((obs, constraint), reward, terminated, truncated) from given action
    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool);
    /// returns the device of current environment
    /// the devices which the training loop uses must be fixed with one device. Do not use more than two devices.
    fn device(&self) -> Device;
}

pub mod cartpole;
pub use cartpole::CartPole;

pub mod tape;
pub use tape::Tape;

use crate::data::Batchable;