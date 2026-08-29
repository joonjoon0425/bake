//! A trait for head which produces state values and basic implementations
use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig}, tensor::Device};

use crate::exploration::{NoiseReset, NoisyLinear};

/// head which produces state values
pub trait VHead: AutodiffModule + Clone + ModuleDisplay {
    /// get the state value from encoded observation
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1>;
}

/// basic linear value head
#[derive(Module, Debug)]
pub struct LinearVHead {
    layer: Linear
}

impl LinearVHead {
    /// create a new value head
    pub fn new(d_input: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, 1).init(device)
        }
    }
}

impl VHead for LinearVHead {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1> {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}

/// A Value head for NoisyNet
#[derive(Module, Debug)]
pub struct NoisyVHead {
    layer: NoisyLinear,
}

impl NoisyVHead {
    /// create a new noisy value head
    pub fn new(d_input: usize, device: &Device) -> Self {
        Self { layer: NoisyLinear::new(d_input, 1, device) }
    }
}

impl NoiseReset for NoisyVHead {
    fn reset_noise(&mut self) { self.layer.reset_noise(); }
}

impl VHead for NoisyVHead {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1> {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}