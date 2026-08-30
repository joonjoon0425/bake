//! A Basic implementations of networks
//! 
pub mod qnet;
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::{Activation, ActivationConfig}}, tensor::Device};
pub use qnet::*;

pub mod dueling_qnet;
pub use dueling_qnet::*;

pub mod policy_net;
pub use policy_net::*;

pub mod actorcritic_net;
pub use actorcritic_net::*;

use crate::exploration::{NoiseReset, NoisyLinear};

#[derive(Module, Debug)]
struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

impl Mlp {
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 1 { panic!("Mlp requires at least one dim"); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(LinearConfig::new(dim, dims[i + 1]).init(device))
        }
        
        Self {
            layers,
            activation: activation.init(device),
        }
    }
    
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}

#[derive(Module, Debug)]
struct NoisyMlp {
    layers: Vec<NoisyLinear>,
    activation: Activation,
}

impl NoisyMlp {
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 1 { panic!("NoisyMlp requires at least one dim"); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(NoisyLinear::new(dim, dims[i + 1], device));
        }
        
        Self {
            layers,
            activation: activation.init(device),
        }
    }
    
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        let mut x = obs;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}

impl NoiseReset for NoisyMlp {
    fn reset_noise(&mut self) {
        for layer in &mut self.layers {
            layer.reset_noise();
        }
    }
}