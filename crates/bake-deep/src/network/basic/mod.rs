//! A Basic implementations of networks
//! 
pub mod mlp_qnet;
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::{Activation, ActivationConfig}}, tensor::Device};
pub use mlp_qnet::*;

pub mod mlp_dueling_qet;
pub use mlp_dueling_qet::*;

pub mod mlp_policy_net;
pub use mlp_policy_net::*;

pub mod mlp_actorcritic_net;
pub use mlp_actorcritic_net::*;

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
