//! Strategies for exploration
//! 
use bake_rl_core::constraint::Constraint;

/// A basic trait for all exploration strategies
pub trait Exploration {
    /// Computes a probability of given action to happen in given state
    fn prob<C: Constraint>(&self, qvalue: Vec<f32>, action: usize, constraint: C) -> f32;
    /// Picks an action from given action values, with constraint if available
    fn sample<C: Constraint>(&mut self, qvalue: Vec<f32>, constraint: C) -> usize;
}

pub mod greedy;
pub use greedy::Greedy;

pub mod eps_greedy;
pub use eps_greedy::EpsGreedy;

pub mod boltzmann;
pub use boltzmann::Boltzmann;