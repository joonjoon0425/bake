use crate::core::{data::Batchable, env::space::Space};
use burn::prelude::*;

/// `VectorEnvironment` trait which all vector environments must implement
pub trait VectorEnvironment {
    /// type of observation space
    type ObsSpace: Space<Element: Batchable>;
    /// type of action space
    type ActionSpace: Space<Element: Batchable>;
    /// type of constraint
    type Constraint: Batchable;
    /// type of reward
    type Reward: Batchable;
    /// type of additional information, which is returned every `step` and `reset`
    type Info: Clone;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environments, according to given mask
    /// - 0: does not reset
    /// - 1: reset
    fn reset(&mut self, mask: Tensor<1, Bool>) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info);
    /// return the number of environments
    fn n_envs(&self) -> usize;
}

/// configuration trait for `VectorizedEnvironment`
pub trait VectorizedEnvironmentConfig {
    /// the environment which configuration is about
    type Environment: VectorEnvironment;
    /// create a new environment
    fn init(self) -> Self::Environment;
}