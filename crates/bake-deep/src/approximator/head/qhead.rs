//! A Q Value Head trait and basic implementations

use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig}, tensor::Device};
use crate::{constraint::DiscreteConstraint, exploration::{NoiseReset, NoisyLinear}};

/// head which produces q values
pub trait QHead: AutodiffModule + Clone + ModuleDisplay {
    /// get the q values from encoded observation
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2>;
}

/// basic linear q value head
#[derive(Module, Debug)]
pub struct LinearQHead {
    layer: Linear
}

impl LinearQHead {
    /// create a new LinearQHead
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl NoiseReset for LinearQHead {}

impl QHead for LinearQHead {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let x = self.layer.forward(encoded);
        constraint.apply(x, -1e9)
    }
}

/// a head for dueling methods
#[derive(Module, Debug)]
pub struct LinearDuelingQHead {
    value: Linear,
    advantage: Linear,
}

impl LinearDuelingQHead {
    /// create a new LinearDuelingQHead
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            value: LinearConfig::new(d_input, 1).init(device),
            advantage: LinearConfig::new(d_input, d_output).init(device),
        }
    }
}

impl NoiseReset for LinearDuelingQHead {}

impl QHead for LinearDuelingQHead {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let v = self.value.forward(encoded.clone());
        let a = self.advantage.forward(encoded);
        let mean = constraint.clone().mean_dim(1, a.clone());
        constraint.apply(v + a - mean, -1e9)
    }
}

/// Noisy Q Head for NoisyNet
#[derive(Module, Debug)]
pub struct NoisyQHead {
    layer: NoisyLinear,
}

impl NoisyQHead {
    /// create a new NoisyQHead
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self { layer: NoisyLinear::new(d_input, d_output, device) }
    }
}

impl NoiseReset for NoisyQHead {
    fn reset_noise(&mut self) { self.layer.reset_noise(); }
}

impl QHead for NoisyQHead {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let x = self.layer.forward(encoded);
        constraint.apply(x, -1e9)
    }
}

/// a head for noisy dueling methods
#[derive(Module, Debug)]
pub struct NoisyDuelingQHead {
    value: NoisyLinear,
    advantage: NoisyLinear,
}

impl NoisyDuelingQHead {
    /// create a new NoisyDuelingQHead
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            value: NoisyLinear::new(d_input, 1, device),
            advantage: NoisyLinear::new(d_input, d_output, device),
        }
    }
}

impl NoiseReset for NoisyDuelingQHead {
    fn reset_noise(&mut self) { self.value.reset_noise(); self.advantage.reset_noise(); }
}

impl QHead for NoisyDuelingQHead {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let v = self.value.forward(encoded.clone());
        let a = self.advantage.forward(encoded);
        let mean = constraint.clone().mean_dim(1, a.clone());
        constraint.apply(v + a - mean, -1e9)
    }
}