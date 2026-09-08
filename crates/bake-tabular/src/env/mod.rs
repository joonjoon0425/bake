//! A rust native environments for tabular rl
//! 

use crate::constraint::Constraint;
/// Basic trait for tabular environments
pub trait Env {
    /// The type of a mask the environment provides  
    /// If the environment does not provide masks, use NoMask<ACTION_NUM>
    type Constraint: Constraint;
    
    /// reset the environment
    fn reset(&mut self) -> (usize, Self::Constraint);
    /// take one step of environment with given action
    /// and returns a step result, which is: next_obs, reward, terminated, truncated, mask
    fn step(&mut self, action: usize) -> (usize, f32, bool, bool, Self::Constraint);
}