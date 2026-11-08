//! LossWeight trait for weighted losses

use burn::prelude::*;

/// traits for weights
pub trait LossWeight: Sized {
    /// apply weights to loss before reduction
    fn apply(self, loss_before_reduction: Tensor<1>) -> Tensor<1>;
    /// enable autodiff
    fn autodiff(self) -> Self { self }
}

impl LossWeight for () {
    fn apply(self, loss_before_reduction: Tensor<1>) -> Tensor<1> { loss_before_reduction }
}