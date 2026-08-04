// The basic trait for environments
pub trait Environment {
    type Obs;
    type Action;

    fn step(&mut self, action: Self::Action) -> (Self::Obs, f32, bool, bool);
    fn reset(&mut self) -> Self::Obs;
}

pub mod cartpole;
pub use cartpole::*;