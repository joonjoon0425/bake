use crate::core::data::BatchOps;
use crate::core::env::space::BatchedSpace;
use burn::prelude::*;

/// `BatchedEnvironment` trait which produces batched data
/// - Note that the `BatchedEnvironment` has APIs like Farama Foundation's `Gymnasium` vector environment
/// - But basically, this environment is for algorithms which consumes batch data types, so it can be used as single, non-vector environment
pub trait BatchedEnvironment {
    /// type of observation space
    type ObsSpace: BatchedSpace;
    /// type of action space
    type ActionSpace: BatchedSpace;
    /// type of constraint
    type Constraint: BatchOps;
    /// type of reward
    type Reward: BatchOps;
    /// type of additional information, which is returned every `step` and `reset`
    type Info: Clone;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environments, according to given mask. If `None` was given, resets all environments
    /// - 0: does not reset
    /// - 1: reset
    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info);
    /// return the number of environments
    fn n_envs(&self) -> usize;
    /// return the device
    fn device(&self) -> Device;
}

/// configuration trait for `BatchedEnvironment`
pub trait BatchedEnvironmentConfig {
    /// the environment which configuration is about
    type Environment: BatchedEnvironment;
    /// create a new environment
    fn init(self, n_envs: usize, device: &Device) -> Self::Environment;
}