//! Space for discrete elements

use burn::{prelude::*, tensor::Distribution::Uniform};
use crate::core::env::space::Space;

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

    fn sample(&self) -> Self::Element {
        Tensor::random([1], Uniform(0., self.n as f64), &self.device)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}