//! The environment crate for bake-rl
#![warn(missing_docs)]

use crate::space::Space;
pub mod collection;
pub mod tape;
pub mod space;
pub mod wrapper;
/// Environment trait which all environments must implement
/// # Warning
/// - The environment is basically non-vectorzied
pub trait Environment {
    /// the observation space of environment
    type ObsSpace: Space;
    /// the action space of environment
    type ActionSpace: Space;
    /// the constraint of which the environment produces
    type Constraint;

    /// reset the environment
    fn reset(&mut self) -> (<Self::ObsSpace as Space>::Element, Self::Constraint);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated)
    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), f32, bool, bool);

    /// returns the observation space of environment
    fn obs_space(&self) -> &Self::ObsSpace;
    /// returns the action space of environment
    fn action_space(&self) -> &Self::ActionSpace;
}

/// Environment configuration trait
pub trait EnvironmentConfiguration {
    /// the type of environment which the configuration is about
    type Env: Environment;
    /// create a new environment
    fn init(self) -> Self::Env;
}

#[cfg(feature = "deep")]
pub mod vectorized;