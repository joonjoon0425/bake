//! The environment crate for bake-rl
#![warn(missing_docs)]
pub mod collection;
pub mod tape;
/// Environment trait which all environments must implement
/// # Warning
/// - The environment is basically non-vectorzied
pub trait Environment {
    /// the configuration type
    type EnvConfig: EnvironmentConfiguration<Env = Self>;
    /// the state, or observation, of environment
    type Obs;
    /// the type of action in the environment
    type Action;
    /// the constraint of which the environment produces
    type Constraint;
    /// reset the environment
    fn reset(&mut self) -> (Self::Obs, Self::Constraint);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated)
    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool);
    /// gives the configuration of the environment
    fn config(&self) -> &Self::EnvConfig;
}

/// Environment configuration trait
pub trait EnvironmentConfiguration {
    /// the type of environment which the configuration is about
    type Env: Environment<EnvConfig = Self>;
    /// create a new environment
    fn init(self) -> Self::Env;
}
/// Configuration for Tabular environments
pub trait TabularEnvironmentConfiguration : EnvironmentConfiguration<Env: Environment<Obs = usize, Action = usize>> {
    /// total number of observations
    fn n_obs(&self) -> usize;
    /// total number of actions
    fn n_actions(&self) -> usize;
}
/// Configuration for Discrete action environments
pub trait DiscreteEnvironmentConfiguration<const OBS_RANK: usize> : EnvironmentConfiguration {
    /// the shape of the observation
    const OBS_SHAPE: [usize; OBS_RANK];
    /// shape of observation 
    fn obs_shape(&self) -> [usize; OBS_RANK] {
        Self::OBS_SHAPE
    }
    /// range of observation
    fn obs_range(&self) -> (<Self::Env as Environment>::Obs, <Self::Env as Environment>::Obs);
    /// total number of actions
    fn n_actions(&self) -> usize;
}
/// Configuration for Continuous action environments
pub trait ContinuousEnvironmentConfiguration<const OBS_RANK: usize, const ACTION_RANK: usize> : EnvironmentConfiguration {
    /// the shape of the observation
    const OBS_SHAPE: [usize; OBS_RANK];
    /// the shape of the action
    const ACTION_SHAPE: [usize; ACTION_RANK];
    /// shape of observation 
    fn obs_shape(&self) -> [usize; OBS_RANK] {
        Self::OBS_SHAPE
    }
    /// range of observation
    fn obs_range(&self) -> (<Self::Env as Environment>::Obs, <Self::Env as Environment>::Obs);
    /// shape of action 
    fn action_shape(&self) -> [usize; ACTION_RANK] {
        Self::ACTION_SHAPE
    }
    /// range of action
    fn action_range(&self) -> (<Self::Env as Environment>::Action, <Self::Env as Environment>::Action);
}

#[cfg(feature = "deep")]
pub mod vectorized;