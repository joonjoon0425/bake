//! A Q Value Head trait and basic implementations

use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig}, tensor::Device};
use crate::constraint::DiscreteConstraint;

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

impl QHead for LinearDuelingQHead {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let v = self.value.forward(encoded.clone());
        let a = self.advantage.forward(encoded);
        let mean = constraint.clone().mean_dim(1, a.clone());
        constraint.apply(v + a - mean, -1e9)
    }
}