use burn::{prelude::*, tensor::kind::Basic};
use crate::core::data::batch_dim::BatchShape;

/// `burn::Tensor`, with batch dimension
#[derive(Debug, Clone)]
pub struct BatchTensor<const D: usize, K: KindAutoDiff = Float> {
    shape: BatchShape,
    t: Tensor<D, K>,
}

/// for autodiff implementation
pub trait KindAutoDiff: Basic + 'static {
    fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self>;
}
impl KindAutoDiff for Float { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t.autodiff() } }
impl KindAutoDiff for Int   { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }
impl KindAutoDiff for Bool  { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }

mod batch_dim_impl;
mod ops;