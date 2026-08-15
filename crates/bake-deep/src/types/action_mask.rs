//! Action mask trait and implementations for masking

use burn::Tensor;

use crate::types::Batchable;

/// Basic Mask trait which all masks must implement
pub trait ActionMask: Batchable {
    type Value;

    fn apply(batched_mask: <Self as Batchable>::Batched, values: Self::Value, fill_value: f32) -> Self::Value;
}

impl ActionMask for () {
    type Value = Tensor<2>;

    fn apply(_: <Self as Batchable>::Batched, values: Self::Value, _: f32) -> Self::Value { values }
}

#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>(pub [bool; D]);

impl<const D: usize> ActionMask for DiscreteMask<D> {
    type Value = Tensor<2>;

    fn apply(batched_mask: <Self as Batchable>::Batched, value: Self::Value, fill_value: f32) -> Self::Value {
        value.mask_fill(batched_mask, fill_value)
    }
}