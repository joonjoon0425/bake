//! Space for discrete elements

use burn::{prelude::*, tensor::Distribution::Uniform};
use crate::core::data::BatchOps;

use super::Space;

/// Discrete space
#[derive(Debug, Clone)]
pub struct DiscreteSpace {
    n: usize,
    device: Device
}

impl DiscreteSpace {
    /// create a new `DiscreteSpace`
    pub fn new(n: usize, device: &Device) -> Self {
        Self { n, device: device.clone() }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl Space for DiscreteSpace {
    type Element = Tensor<1, Int>;

    fn sample(&self, num: usize) -> Self::Element {
        let vec = 
            (0..num).into_iter().map(|_| Tensor::random([1], Uniform(0., self.n as f64), &self.device)).collect();
        BatchOps::cat_batch(&vec![vec![1]; num], 0, vec)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}