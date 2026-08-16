//! # Environments for Deep RL
use crate::types::{ActionMask, Batchable};

/// The basic trait for environments
/// #### Warning
/// All environments should return the obs with batch dimension, and recieve action with batch dimension <br>
/// The first principle of this framework is that everything owns a batch dimension<br>
/// This is due to the fixed rank of burn's `Tensor`s
pub trait Env {
    type Obs: Batchable;
    type Action: Batchable;
    type Mask: ActionMask;

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Mask), f32, bool, bool);
    fn reset(&mut self) -> (Self::Obs, Self::Mask);
}

pub mod cartpole;
pub use cartpole::*;