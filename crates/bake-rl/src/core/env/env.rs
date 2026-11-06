//! Trait for environment

use crate::core::env::space::NativeSpace;
/// `Environment` trait which all non-vectorized environments must implement
/// - For custom non-vetorized environment, implement this trait
/// - For custom vetorized environment, implement `VectorEnvironment` trait
pub trait Environment {
    /// type of observation space
    type ObsSpace: NativeSpace;
    /// type of action space
    type ActionSpace: NativeSpace;
    /// type of constraint
    type Constraint: Clone;
    /// type of reward
    type Reward: Clone;
    /// type of additional information, which is returned every `step` and `reset`
    type Info: Clone;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environment and return the tuple ((obs, constraint), information)
    fn reset(&mut self) -> ((<Self::ObsSpace as NativeSpace>::Element, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <Self::ActionSpace as NativeSpace>::Element) -> ((<Self::ObsSpace as NativeSpace>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info);
}

/// the configuration trait for environments
pub trait EnvironmentConfig {
    /// the environment which the configuration is about
    type Environment: Environment;

    /// create a new environment, with given device
    fn init(self) -> Self::Environment;
}