//! A simple MLP Dueling QNet
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::ActivationConfig}, tensor::Device};

use crate::network::{DuelingQNet, basic::Mlp};

/// A Simple MLP Dueling QNet
#[derive(Module, Debug)]
pub struct MlpDuelingQNet {
    mlp: Mlp,
    advantage_layer: Linear,
    value_layer: Linear,
}

impl MlpDuelingQNet {
    /// create a new MlpDuelingQNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpDuelingQNet requires at least two dims: input dimension and the number of actions."); }
        let mlp = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let advantage_layer = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        let value_layer = LinearConfig::new(dims[dims.len() - 2], 1).init(device);
        
        Self {
            mlp,
            advantage_layer,
            value_layer,
        }
    }
}

impl DuelingQNet for MlpDuelingQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>) {
        let x = self.mlp.forward(obs);
        let value = self.value_layer.forward(x.clone()).squeeze_dim(1);
        let advantage = self.advantage_layer.forward(x);
        (value, advantage)
    }
}