use crate::types::*;

pub trait Env {
    type Mask : Mask;
    
    fn reset(&mut self) -> usize;
    fn step(&mut self, action: usize) -> Step;
    fn action_mask(&self) -> Self::Mask;
}

pub mod grid_world;
pub use grid_world::*;