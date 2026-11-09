//! greedy action selector
use burn::prelude::*;
use rand::prelude::*;
use crate::core::constraint::{BatchedDiscreteConstraint, DiscreteConstraint};

use super::DiscreteActionSelector;
use super::BatchedDiscreteActionSelector;

/// A greedy policy implementation
pub struct Greedy;
/// A greedy policy implementation
pub struct BatchedGreedy;

impl DiscreteActionSelector for Greedy {
    fn sample<C: DiscreteConstraint, R: Rng>(&self, _rng: &mut R, qvalues: &[f32], constraint: C) -> usize {
        let qvalues = constraint.apply(qvalues, -1e9);
        super::argmax_max(&qvalues).0
    }
}

impl BatchedDiscreteActionSelector for BatchedGreedy {
    fn sample<C: BatchedDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let qvalues = constraint.apply(qvalues, -1e9);
        qvalues.argmax(1).squeeze_dim(1)
    }
}