//! A trait for head which produces state values and basic implementations
use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig}, tensor::Device};

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
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl VHead for LinearVHead {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1> {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}