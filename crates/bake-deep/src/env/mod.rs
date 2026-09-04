//! A native rust environments for deep rl
//! 

/// ### A trait which all environments must implement
/// ##### Warning
/// All environments should return the obs with batch dimension, and receive action with batch dimension.
/// The first principle of this framework is that everything owns a batch dimension.
/// This is due to the fixed rank of burn's `Tensor`.
pub trait Environment {
    /// The observation type which environment produces
    type Obs;
    /// The action type which environment receives
    type Action;
    /// The constraint for actions. If the environment does not provides constraint, set it as `Unconstrained`
    type Constraint;

    /// reset the environment
    fn reset(&mut self) -> (Self::Obs, Self::Constraint);
    /// take one step and return a tuple of ((obs, constraint), reward, terminated, truncated) from given action
    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool);
}