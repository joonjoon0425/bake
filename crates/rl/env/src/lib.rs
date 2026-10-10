//! The environment crate for bake-rl
#![warn(missing_docs)]
pub mod collection;
pub mod tape;
pub mod wrapper;
/// Environment trait which all environments must implement
/// # Warning
/// - The environment is basically non-vectorzied
pub trait Environment {
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
}

/// Trait for Tabular environments
pub trait TabularEnvironment : Environment<Obs = usize, Action = usize> + Sized {
    /// total number of observations
    fn n_obs(&self) -> usize;
    /// total number of actions
    fn n_actions(&self) -> usize;
}
/// Trait for Discrete action environments
pub trait DiscreteEnvironment : Environment<Action = usize> {
    /// shape of observation 
    fn obs_shape(&self) -> Vec<usize>;
    /// range of observation
    fn obs_range(&self) -> (Self::Obs, Self::Obs);
    /// total number of actions
    fn n_actions(&self) -> usize;
}
/// Trait for Continuous action environments
pub trait ContinuousEnvironment : Environment {
    /// shape of observation 
    fn obs_shape(&self) -> Vec<usize>;
    /// range of observation
    fn obs_range(&self) -> (Self::Obs, Self::Obs);
    /// shape of action 
    fn action_shape(&self) -> Vec<usize>;
    /// range of action
    fn action_range(&self) -> (Self::Action, Self::Action);
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