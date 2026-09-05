//! Strategies for exploration
//! 
use burn::prelude::*;
use crate::{constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction};

/// A trait which all exploration strategies must implement
pub trait Exploration {
    /// sample an action using given q-function, observation and constraint
    fn sample<Q: DiscreteQFunction>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int>;
}

pub mod greedy;
pub use greedy::Greedy;

pub mod eps_greedy;
pub use eps_greedy::EpsGreedy;

pub mod boltzmann;
pub use boltzmann::Boltzmann;