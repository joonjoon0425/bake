//! exploration traits and implementations
use burn::prelude::*;
use rand::prelude::*;
use crate::core::constraint::{BatchedDiscreteConstraint, DiscreteConstraint};

/// A trait which all 'non-batch' exploration strategies must implement
pub trait DiscreteActionExploration {
    /// sample an action using given discrete action values, with given constraint
    fn sample<C: DiscreteConstraint, R: Rng>(&self, rng: &mut R, qvalues: &[f32], constraint: C) -> usize;
}

/// A trait which all 'batch' exploration strategies must implement
pub trait BatchedDiscreteActionExploration {
    /// sample an action using given discrete action values, with given constraint
    fn sample<C: BatchedDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int>;
}

pub mod greedy;
pub use greedy::{Greedy, BatchedGreedy};

pub mod epsilon_greedy;
pub use epsilon_greedy::{EpsilonGreedy, BatchedEpsilonGreedy};

pub mod boltzmann;
pub use boltzmann::{Boltzmann, BatchedBoltzmann};

/// utility max function with index
fn argmax_max(values: &[f32]) -> (usize, f32) {
    let mut max = f32::MIN;
    let mut idx = 0;

    for (i, &v) in values.iter().enumerate() {
        if max < v {
            max = v;
            idx = i;
        }
    }
    (idx, max)
}