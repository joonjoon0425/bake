//! # Environments for tabular algorithms
use crate::types::*;

/// Basic trait for tabular environments
pub trait Env {
    /// The type of a mask the environment provides  
    /// If the environment does not provide masks, use NoMask<ACTION_NUM>
    type Mask : Mask;
    
    /// reset the environment
    fn reset(&mut self) -> usize;
    /// go ahead one step with given action
    /// returns a step result, which is: next_obs, reward, terminated, truncated, mask
    fn step(&mut self, action: usize) -> Step;
    /// deprecated, should be deleted
    fn action_mask(&self) -> Self::Mask;
}

pub mod grid_world;
pub use grid_world::*;