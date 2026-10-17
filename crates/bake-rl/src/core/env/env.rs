//! Trait for environment

use crate::core::env::space::{Space, switch::ToBatched};

/// `Environment` trait which all non-vectorized environments must implement
/// # Warning
/// - If you are using `burn::Tensor` for observation space, action and reward, the environment must have a batch dimension for its outputs (obs, action, constraint, reward, except for the flags), even if it is implemented as non-vectorized environment
/// - This is due to the fixed rank of `burn`'s `Tensor`s.
/// - Otherwise, batch dimension is not required
pub trait Environment {
    /// type of observation space
    type ObsSpace: Space;
    /// type of action space
    type ActionSpace: Space;
    /// type of constraint
    type Constraint: ToBatched;
    /// type of reward
    type Reward: ToBatched;
    /// type of additional information, which is returned every `step` and `reset`
    type Info;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environment and return the tuple ((obs, constraint), information)
    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info);

    /// return a new configuration struct
    fn config() -> impl EnvironmentConfig<Environment = Self>;
}

/// the configuration trait for environments
pub trait EnvironmentConfig {
    /// the environment which the configuration is about
    type Environment: Environment;

    /// create a new environment
    fn init(self) -> Self::Environment;
}