//! A simple MLP QNet
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::ActivationConfig}, tensor::Device};

use crate::{exploration::{NoiseReset, NoisyLinear}, network::{QNet, basic::{Mlp, NoisyMlp}}};

/// A Simple MLP QNet
#[derive(Module, Debug)]
pub struct MlpQNet {
    mlp: Mlp,
    layer: Linear,
}

impl MlpQNet {
    /// create a new MlpQNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpQNet requires at least two dims: input dimension and the number of actions."); }
        let mlp = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let layer = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        Self {
            mlp,
            layer,
        }
    }
}

impl QNet for MlpQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let x = self.mlp.forward(obs);
        self.layer.forward(x)
    }
}

/// A noisy Mlp QNetwork with NoisyLinear
#[derive(Module, Debug)]
pub struct NoisyMlpQNet {
    mlp: NoisyMlp,
    layer: NoisyLinear,
}

impl NoisyMlpQNet {
    /// create a new NoisyMlpQNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpQNet requires at least two dims: input dimension and the number of actions."); }
        let mlp = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);
        let layer = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        Self {
            mlp,
            layer,
        }
    }
}

impl QNet for NoisyMlpQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let x = self.mlp.forward(obs);
        self.layer.forward(x)
    }
}

impl NoiseReset for NoisyMlpQNet {
    fn reset_noise(&mut self) {
        self.mlp.reset_noise();
        self.layer.reset_noise();
    }
}