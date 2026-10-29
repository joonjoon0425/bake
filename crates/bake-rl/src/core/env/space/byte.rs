//! Byte (u8) space

use crate::core::data::{AttachBatchDim, BatchTensor, WithBatchDim};

use super::Space;
use burn::prelude::*;

/// Byte space with feature rank (D - 1)
#[derive(Debug, Clone)]
pub struct ByteSpace<const D: usize> {
    device: Device,
    shape: Vec<usize>,
}

impl<const D: usize> ByteSpace<D> {
    /// create a new byte space, with feature rank (D - 1)
    /// # Warning
    /// - note that D is (feature rank + 1)
    /// - this is due to the fixed rank of `burn::Tensor`
    pub fn new(shape: &[usize], device: &Device) -> Self {
        Self { shape: shape.try_into().unwrap(), device: device.clone() }
    }
    /// return the shape
    /// # Warning
    /// - Due to fixed rank of `burn::Tensor`, shape will always have dimension 1 in the first dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<const D: usize> Space for ByteSpace<D> {
    type Element = BatchTensor<D, Int>;

    fn sample(&self, num: usize) -> Self::Element {
        let vec = (0..num).into_iter()
            .map(|_| {
                let s = Tensor::random(self.shape(), burn::tensor::Distribution::Uniform(0., 255.), (&self.device, burn::tensor::DType::U8));
                s.attach_empty_dim()
            })
            .collect();
        WithBatchDim::stack(vec)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}