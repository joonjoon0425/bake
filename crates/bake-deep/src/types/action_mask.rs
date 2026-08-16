//! Action mask trait and implementations for masking
use burn::{Tensor, tensor::{Bool}};
/// Basic Mask trait which all masks must implement
pub trait ActionMask: Clone {
    type Value;
    fn apply(self, values: Self::Value, fill_value: f32) -> Self::Value;
    fn mean_dim(self, dim: i64, values: Self::Value) -> Self::Value;
}
/// Basic Discrete action mask for tensors
#[derive(Debug, Clone)]
pub struct DiscreteMask<const ACTION_NUM: usize>(pub Tensor<1, Bool>);
impl<const ACTION_NUM: usize> ActionMask for DiscreteMask<ACTION_NUM> {
    type Value = Tensor<1>;

    fn apply(self, values: Self::Value, fill_value: f32) -> Self::Value {
        values.mask_fill(self.0.bool_not(), fill_value)
    }

    fn mean_dim(self, dim: i64, values: Self::Value) -> Self::Value {
        let invalid = self.clone().0.bool_not();
        let n_possible_actions = self.0.float().sum_dim(dim);
        let mean = values.mask_fill(invalid.clone(), 0f32) / n_possible_actions;
        mean
    }
}

/// Batched Discrete Action mask
#[derive(Debug, Clone)]
pub struct BatchedDiscreteMask<const ACTION_NUM: usize>(pub Tensor<2, Bool>);
impl<const ACTION_NUM: usize> ActionMask for BatchedDiscreteMask<ACTION_NUM> {
    type Value = Tensor<2>;

    fn apply(self, values: Self::Value, fill_value: f32) -> Self::Value {
        values.mask_fill(self.0.bool_not(), fill_value)
    }

    fn mean_dim(self, dim: i64, values: Self::Value) -> Self::Value {
        let invalid = self.clone().0.bool_not();
        let n_possible_actions = self.0.float().sum_dim(dim);
        let mean = values.mask_fill(invalid.clone(), 0f32) / n_possible_actions;
        mean
    }
}

/// No mask
#[derive(Debug, Clone)]
pub struct NoMask;
impl ActionMask for NoMask {
    type Value = Tensor<1>;
    fn apply(self, values: Self::Value, _: f32) -> Self::Value { values }
    fn mean_dim(self, dim: i64, values: Self::Value) -> Self::Value { values.mean_dim(dim) }
}

#[derive(Debug, Clone)]
pub struct BatchedNoMask;
impl ActionMask for BatchedNoMask {
    type Value = Tensor<1>;
    fn apply(self, values: Self::Value, _: f32) -> Self::Value { values }
    fn mean_dim(self, dim: i64, values: Self::Value) -> Self::Value { values.mean_dim(dim) }
}