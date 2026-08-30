//! A simple MLP Separated ActorCritic
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::ActivationConfig}, tensor::Device};

use crate::network::{ActorCriticNet, EncoderType::{self, Separated}, basic::Mlp};

/// A Simple MLP Separated ActorCritic
#[derive(Module, Debug)]
pub struct MlpActorCriticNet {
    actor_mlp: Mlp,
    critic_mlp: Mlp,
    actor_layer: Linear,
    critic_layer: Linear,
}

impl MlpActorCriticNet {
    /// create a new MlpPolicy struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpActorCriticNet requires at least two dims: input dimension and output dimension."); }
        let actor_mlp = Mlp::new(&dims[..dims.len() - 1], activation.clone(), device);
        let critic_mlp = Mlp::new(&dims[..dims.len() - 1], activation, device);

        let actor_layer = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        let critic_layer = LinearConfig::new(dims[dims.len() - 2], 1).init(device);
        
        Self {
            actor_mlp,
            critic_mlp,
            actor_layer,
            critic_layer,
        }
    }
}

impl ActorCriticNet for MlpActorCriticNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;
    
    fn params(&self, obs: Self::Obs) -> Self::Params {
        self.actor_layer.forward(self.actor_mlp.forward(obs))
    }

    fn values(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic_layer.forward(self.critic_mlp.forward(obs)).squeeze_dim(1)
    }

    fn encoder_type(&self) -> EncoderType {
        Separated
    }
}