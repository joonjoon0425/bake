use crate::traits::Batchable;

// The basic trait for environments
pub trait Environment {
    type Obs: Batchable;
    type Action: Batchable;
    type Mask: Batchable;

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Option<Self::Mask>), f32, bool, bool);
    fn reset(&mut self) -> (Self::Obs, Option<Self::Mask>);
}

pub mod cartpole;
pub use cartpole::*;

