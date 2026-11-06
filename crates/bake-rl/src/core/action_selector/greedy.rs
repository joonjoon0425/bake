//! greedy action selector
use burn::prelude::*;

use crate::core::constraint::{BurnDiscreteConstraint, NativeDiscreteConstraint};

use super::NativeDiscreteActionSelector;
use super::BurnDiscreteActionSelector;

/// A greedy policy implementation
pub struct Greedy;

impl NativeDiscreteActionSelector for Greedy {
    fn sample<C: NativeDiscreteConstraint>(&mut self, qvalues: &[f32], constraint: C) -> usize {
        let qvalues = constraint.apply(qvalues, -1e9);
        super::argmax_max(&qvalues).0
    }
}

impl BurnDiscreteActionSelector for Greedy {
    fn sample<C: BurnDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let qvalues = constraint.apply(qvalues, -1e9);
        qvalues.argmax(1).squeeze_dim(1)
    }
}