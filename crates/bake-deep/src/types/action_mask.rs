//! Action mask trait and implementations for masking
use burn::{Tensor, tensor::{Bool}};
/// Basic Discrete action mask for tensors
#[derive(Debug, Clone)]
pub struct DiscreteMask<const D: usize = 2>(pub Tensor<D, Bool>);
impl<const D: usize> DiscreteMask<D> {
    pub fn apply(self, values: Tensor<D>, fill_value: f32) -> Tensor<D> {
        values.mask_fill(self.0.bool_not(), fill_value)
    }

    pub fn mean_dim(self, dim: usize, values: Tensor<D>) -> Tensor<D> {
        let invalid = self.clone().0.bool_not();
        let n_possible_actions = self.0.float().sum_dim(dim);
        let mean = values.mask_fill(invalid.clone(), 0f32).sum_dim(dim) / n_possible_actions;
        mean
    }
}