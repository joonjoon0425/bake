use crate::types::{ActionMask, Step};

pub trait Env {
    fn reset(&mut self) -> usize;
    fn step(&mut self, action: usize) -> Step;
    fn action_mask(&self) -> Option<ActionMask>;
}

pub mod grid_world;
pub use grid_world::*;