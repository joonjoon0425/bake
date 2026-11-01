//! exploration traits and implementations
use burn::prelude::*;
/// A trait which all exploration strategies must implement
pub trait Exploration {
    /// sample an action using given discrete action values, with given constraint
    fn sample<C: DiscreteConstraint>(&mut self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int>;
}

pub mod greedy;
pub use greedy::Greedy;

pub mod eps_greedy;
pub use eps_greedy::EpsGreedy;

pub mod boltzmann;
pub use boltzmann::Boltzmann;

use crate::core::constraint::DiscreteConstraint;