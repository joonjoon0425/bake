//! A simple MLP Policy
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::ActivationConfig}, tensor::Device};

use crate::network::{PolicyNet, basic::Mlp};

/// A Simple MLP Policy
#[derive(Module, Debug)]
pub struct MlpPolicyNet {
    mlp: Mlp,
    layer: Linear,
}

impl MlpPolicyNet {
    /// create a new MlpPolicy struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpPolicyNet requires at least two dims: input dimension and output dimension."); }
        let mlp = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let layer = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        
        Self {
            mlp,
            layer
        }
    }
}

impl PolicyNet for MlpPolicyNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        self.layer.forward(self.mlp.forward(obs))
    }
}