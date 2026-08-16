//! Encoder trait and basic implementations
//! 

use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig, activation::Activation}, tensor::Device};

use crate::types::Batchable;
pub trait Encoder : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

#[derive(Module, Debug)]
pub struct MLPEncoder {
    layers: Vec<Linear>,
    activation: Activation,
}

impl MLPEncoder {
    pub fn new(dims: Vec<usize>, activation: Activation, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MLPEncoder requires at least two dims: input dimension and output dimension."); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(LinearConfig::new(dim, dims[i + 1]).init(device))
        }
        
        Self {
            layers,
            activation,
        }
    }
}

impl Encoder for MLPEncoder {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}