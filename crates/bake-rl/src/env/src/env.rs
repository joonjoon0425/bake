//! The environment trait
//! 

/// Environment trait which all environments must implement
/// # Warning
/// - The environment is basically non-vectorzied
pub trait Environment {
    /// the configuration type
    type EnvConfig: EnvironmentConfiguration;
    /// the state, or observation, of environment
    type Obs;
    /// the type of action in the environment
    type Action;
    /// the constraint of which the environment produces
    type Constraint;

    /// create a new environment configuration
    fn new() -> Self::EnvConfig;
    /// reset the environment
    fn reset(&mut self) -> (Self::Obs, Self::Constraint);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated)
    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool);
}

/// Environment configuration trait
pub trait EnvironmentConfiguration {
    /// the type of environment which the configuration is about
    type Env: Environment;
    /// create a new environment
    fn init(self) -> Self::Env;
}