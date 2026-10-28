//! Space for continuous elements

use crate::core::{data::{BatchTensor, WithBatchDim, batch_dim::convert::AttachBatchDim}, env::space::Space};
use burn::{prelude::*, tensor::Distribution::Uniform};
/// Continuous space with feature rank (D - 1)
/// - E: element type
///     - `burn::Tensor`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<const D: usize> {
    device: Device,
    shape: Vec<usize>,
    low: Tensor<D>,
    high: Tensor<D>,
}

impl<const D: usize> ContinuousSpace<D> {
    /// create a new continuous space, with feature rank (D - 1)
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    pub fn new(shape: &[usize], low: Tensor<D>, high: Tensor<D>, device: &Device) -> Self {
        Self { shape: shape.try_into().unwrap(), low, high, device: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &Tensor<D> {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &Tensor<D> {
        &self.high
    }
}

impl<const D: usize> Space for ContinuousSpace<D> {
    type Element = BatchTensor<D>;

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn sample(&self, num: usize) -> Self::Element {
        let vec = (0..num).into_iter()
            .map(|_| {
                let s = Tensor::random(self.shape(), Uniform(0., 1.), &self.device);
                ((self.high.clone() - self.low.clone()) * s + self.low.clone()).attach_batch_dim(vec![])
            })
            .collect();
        WithBatchDim::stack(vec)
    }
}