//! A simple MLP Encoder
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::{Activation, ActivationConfig}}, tensor::Device};

use crate::{approximator::Encoder, exploration::{NoiseReset, NoisyLinear}};

/// A Simple mlp encoder struct
#[derive(Module, Debug)]
pub struct MlpEncoder {
    layers: Vec<Linear>,
    activation: Activation,
}

impl MlpEncoder {
    /// create a new MlpEncoder struct with given dimensions and activation unit
    pub fn new(dims: Vec<usize>, activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpEncoder requires at least two dims: input dimension and output dimension."); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(LinearConfig::new(dim, dims[i + 1]).init(device))
        }
        
        Self {
            layers,
            activation: activation.init(device),
        }
    }
}

impl Encoder for MlpEncoder {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}
impl NoiseReset for MlpEncoder {}

/// A Simple noisy mlp encoder struct
#[derive(Module, Debug)]
pub struct NoisyMlpEncoder {
    layers: Vec<NoisyLinear>,
    activation: Activation,
}

impl NoisyMlpEncoder {
    /// create a new MlpEncoder struct with given dimensions and activation unit
    pub fn new(dims: Vec<usize>, activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpEncoder requires at least two dims: input dimension and output dimension."); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(NoisyLinear::new(dim, dims[i + 1], device))
        }
        
        Self {
            layers,
            activation: activation.init(device),
        }
    }
}

impl NoiseReset for NoisyMlpEncoder {
    fn reset_noise(&mut self) {
        for layer in &mut self.layers {
            layer.reset_noise();
        }
    }
}

impl Encoder for NoisyMlpEncoder {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}