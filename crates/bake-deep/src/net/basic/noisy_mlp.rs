//! basic Noisy network Mlp implementations

use burn::prelude::*;
use burn::nn::{activation::{ActivationConfig, Activation}};
use crate::net::{ActorCriticNet, DiscreteDuelingQNet, DiscreteQNet, PolicyNet};
use crate::net::layer::{NoiseReset, NoisyLinear};

/// basic NoisyMlp Encoder part.
/// # Warning
/// The activation is applied at last layer, too.<br>
/// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output.
#[derive(Module, Debug)]
pub struct NoisyMlp {
    layers: Vec<NoisyLinear>,
    activation: Activation,
}

impl NoisyMlp {
    /// create a new NoisyMlp
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
    
    /// compute output
    /// # Warning
    /// the activation is applied last layer, too. <br>
    /// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output./// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output.
    pub fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
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


/// DiscreteQNet implementation with NoisyMlp
#[derive(Module, Debug)]
pub struct NoisyMlpDiscreteQNet {
    encoder: NoisyMlp,
    head: NoisyLinear,
}

impl NoisyMlpDiscreteQNet {
    /// create a new `NoisyMlpDiscreteQNet`
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpDiscreteQNet requires at least two dims: input dimension and the number of actions."); }
        let encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);
        let head = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        Self {
            encoder,
            head,
        }
    }
}

impl DiscreteQNet for NoisyMlpDiscreteQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let x = self.encoder.forward(obs);
        self.head.forward(x)
    }
}

impl NoiseReset for NoisyMlpDiscreteQNet {
    fn reset_noise(&mut self) {
        self.encoder.reset_noise();
        self.head.reset_noise();
    }
}

/// DiscreteDuelingQNet implementation with NoisyMlp
#[derive(Module, Debug)]
pub struct NoisyMlpDiscreteDuelingQNet {
    encoder: NoisyMlp,
    advantage_layer: NoisyLinear,
    value_layer: NoisyLinear,
}

impl NoisyMlpDiscreteDuelingQNet {
    /// create a new `NoisyMlpDiscreteDuelingQNet` struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpDiscreteDuelingQNet requires at least two dims: input dimension and the number of actions."); }
        let encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);
        let advantage_layer = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        let value_layer = NoisyLinear::new(dims[dims.len() - 2], 1, device);
        
        Self {
            encoder,
            advantage_layer,
            value_layer,
        }
    }
}

impl DiscreteDuelingQNet for NoisyMlpDiscreteDuelingQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>) {
        let x = self.encoder.forward(obs);
        let value = self.value_layer.forward(x.clone()).squeeze_dim(1);
        let advantage = self.advantage_layer.forward(x);
        (value, advantage)
    }
}

/// PolicyNet implementation with NoisyMlp
#[derive(Module, Debug)]
pub struct NoisyMlpPolicyNet {
    encoder: NoisyMlp,
    head: NoisyLinear,
}

impl NoisyMlpPolicyNet {
    /// create a new `NoisyMlpPolicy` struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpPolicyNet requires at least two dims: input dimension and output dimension."); }
        let encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);
        let head = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        
        Self {
            encoder,
            head
        }
    }
}

impl PolicyNet for NoisyMlpPolicyNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        self.head.forward(self.encoder.forward(obs))
    }
}

/// Separated ActorCritic implementation with NoisyMlp
#[derive(Module, Debug)]
pub struct NoisyMlpSeparatedActorCriticNet {
    actor_encoder: NoisyMlp,
    critic_encoder: NoisyMlp,
    actor_head: NoisyLinear,
    critic_head: NoisyLinear,
}

impl NoisyMlpSeparatedActorCriticNet {
    /// create a new NoisyMlpActorCriticNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpSeparatedActorCriticNet requires at least two dims: input dimension and output dimension."); }
        let actor_encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation.clone(), device);
        let critic_encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);

        let actor_head = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        let critic_head = NoisyLinear::new(dims[dims.len() - 2], 1, device);
        
        Self {
            actor_encoder,
            critic_encoder,
            actor_head,
            critic_head,
        }
    }
}

impl ActorCriticNet for NoisyMlpSeparatedActorCriticNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;
    
    fn params(&self, obs: Self::Obs) -> Self::Params {
        self.actor_head.forward(self.actor_encoder.forward(obs))
    }

    fn values(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic_head.forward(self.critic_encoder.forward(obs)).squeeze_dim(1)
    }

    fn encoder_type(&self) -> crate::contract::actor_critic::EncoderType {
        crate::contract::actor_critic::EncoderType::Separated
    }
}

/// Shared ActorCritic implementation with NoisyMlp
#[derive(Module, Debug)]
pub struct NoisyMlpSharedActorCriticNet {
    encoder: NoisyMlp,
    actor_head: NoisyLinear,
    critic_head: NoisyLinear,
}

impl NoisyMlpSharedActorCriticNet {
    /// create a new `NoisyMlpSharedActorCriticNet` struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("NoisyMlpSharedActorCriticNet requires at least two dims: input dimension and output dimension."); }
        let encoder = NoisyMlp::new(&dims[..dims.len() - 1], activation, device);

        let actor_head = NoisyLinear::new(dims[dims.len() - 2], dims[dims.len() - 1], device);
        let critic_head = NoisyLinear::new(dims[dims.len() - 2], 1, device);
        
        Self {
            encoder,
            actor_head,
            critic_head,
        }
    }
}

impl ActorCriticNet for NoisyMlpSharedActorCriticNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;
    
    fn params(&self, obs: Self::Obs) -> Self::Params {
        self.actor_head.forward(self.encoder.forward(obs))
    }

    fn values(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic_head.forward(self.encoder.forward(obs)).squeeze_dim(1)
    }

    fn forward(&self, obs: Self::Obs) -> (Self::Params, Tensor<1>) {
        let encoded = self.encoder.forward(obs);
        (self.actor_head.forward(encoded.clone()), self.critic_head.forward(encoded).squeeze_dim(1))
    }

    fn encoder_type(&self) -> crate::contract::actor_critic::EncoderType {
        crate::contract::actor_critic::EncoderType::Shared
    }
}