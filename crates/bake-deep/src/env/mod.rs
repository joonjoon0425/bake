//! # Environments for Deep RL
use crate::types::Batchable;

// The basic trait for environments
pub trait Env {
    type Obs: Batchable;
    type Action: Batchable;
    type Mask: Batchable;

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Mask), f32, bool, bool);
    fn reset(&mut self) -> (Self::Obs, Self::Mask);
}

pub mod cartpole;
pub use cartpole::*;