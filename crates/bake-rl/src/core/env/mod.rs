//! RL environment trait

pub mod space;
pub use space::Space;

/// Basic environment trait
pub trait Environment {
    /// the type of observation space
    type ObsSpace: Space;
    /// the type of action space
    type ActionSpace: Space;
    /// the type of constraint
    type Constraint;
    /// the type of reward
    type Reward;
    /// the type of `terminated` and `truncated` flag
    type Flag;
    /// additional information which is given every `step` and `reset`
    type Info;

    /// reset the environment
    fn reset(&mut self) -> (<Self::ObsSpace as Space>::Element, Self::Constraint, Self::Info);
    /// take one step with given action and returns (obs, constraint, reward, terminated, truncated, info) tuple
    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> (<Self::ObsSpace as Space>::Element, Self::Constraint, Self::Reward, Self::Flag, Self::Flag, Self::Info);
    /// return the reference of observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the reference of action space
    fn action_space(&self) -> &Self::ActionSpace;
}

/// Environment configuration trait
pub trait EnvironmentConfig {
    /// the type of environment which the configuration is about
    type Env: Environment;
    /// create a new environment
    fn init(self) -> Self::Env;
}
