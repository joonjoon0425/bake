//! Space for continuous elements

use crate::core::{data::{WithBatchDim, batch_dim::convert::AttachBatchDim}, env::space::Space};
use burn::{prelude::*, tensor::Distribution::Uniform};
/// Continuous space
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
    /// create a new continuous space, with feature rank D
    pub fn new(shape: &[usize], low: Tensor<D>, high: Tensor<D>, device: &Device) -> Self {
        Self { shape: shape.try_into().unwrap(), low, high, device: device.clone() }
    }
    /// return the shape
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
    type Element = Tensor<D>;

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn sample(&self, num: usize) -> <Self::Element as AttachBatchDim>::Attached {
        let vec = (0..num).into_iter()
            .map(|_| {
                let s = Tensor::random(self.shape(), Uniform(0., 1.), &self.device);
                ((self.low.clone() - self.high.clone()) * s + self.low.clone()).attach_batch_dim(vec![])
            })
            .collect();
        WithBatchDim::stack(vec)
    }
}