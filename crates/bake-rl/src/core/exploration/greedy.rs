//! greedy policy implementation
use burn::prelude::*;
use crate::core::constraint::DiscreteConstraint;

use super::Exploration;

/// A greedy policy implementation
pub struct Greedy;

impl Exploration for Greedy {
    fn sample<C: DiscreteConstraint>(&mut self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let qvalues = constraint.apply(qvalues, -1e9);
        qvalues.argmax(1).squeeze_dim(1)
    }
}