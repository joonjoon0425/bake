//! Action mask trait and implementations for masking
use std::fmt::Debug;
use burn::{Tensor, module::Module, tensor::Bool};
use crate::types::Batchable;

pub trait DiscreteConstraint<const D: usize = 2> : Debug + Clone + Batchable {
    fn apply(self, values: Tensor<D>, fill_value: f32) -> Tensor<D>;
    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D>;
}

/// Basic Discrete action mask for tensors
#[derive(Debug, Clone)]
pub struct DiscreteMask<const D: usize = 2>(pub Tensor<D, Bool>);
impl<const D: usize> DiscreteConstraint<D> for DiscreteMask<D> {
    fn apply(self, values: Tensor<D>, fill_value: f32) -> Tensor<D> {
        values.mask_fill(self.0.bool_not(), fill_value)
    }

    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D> {
        let invalid = self.clone().0.bool_not();
        let n_possible_actions = self.0.float().sum_dim(dim);
        let mean = values.mask_fill(invalid.clone(), 0f32).sum_dim(dim) / n_possible_actions;
        mean
    }
}

/// unconstrainted obect
#[derive(Module, Debug)]
pub struct Unconstrained;
impl<const D: usize> DiscreteConstraint<D> for Unconstrained {
    fn apply(self, values: Tensor<D>, _: f32) -> Tensor<D> { values }
    fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D> { values.mean_dim(dim) }
}