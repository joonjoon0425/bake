//! A simple MLP Dueling QNet
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::ActivationConfig}, tensor::Device};

use crate::{exploration::{NoiseReset, NoisyLinear}, network::{DuelingQNet, basic::{Mlp, NoisyMlp}}};

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

/// A Simple Noisy Dueling QNetwork
#[derive(Module, Debug)]
pub struct NoisyMlpDuelingQNet {
    mlp: NoisyMlp,
    advantage_layer: NoisyLinear,
    value_layer: NoisyLinear,
}

impl NoisyMlpDuelingQNet {
    /// create a new NoisyMlpDuelingQNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpDuelingQNet requires at least two dims: input dimension and the number of actions."); }
        let mlp = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);
        let advantage_layer = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        let value_layer = NoisyLinear::new(dims[dims.len() - 2], 1, device);
        
        Self {
            mlp,
            advantage_layer,
            value_layer,
        }
    }
}

impl DuelingQNet for NoisyMlpDuelingQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>) {
        let x = self.mlp.forward(obs);
        let value = self.value_layer.forward(x.clone()).squeeze_dim(1);
        let advantage = self.advantage_layer.forward(x);
        (value, advantage)
    }
}

impl NoiseReset for NoisyMlpDuelingQNet {
    fn reset_noise(&mut self) {
        self.mlp.reset_noise();
        self.advantage_layer.reset_noise();
        self.value_layer.reset_noise();
    }
}