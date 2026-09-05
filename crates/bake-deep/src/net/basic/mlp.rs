//! basic Mlp network implementations
use burn::prelude::*;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, activation::{Activation, ActivationConfig}};

use crate::net::{ActorCriticNet, DiscreteDuelingQNet, DiscreteQNet, PolicyNet};

/// basic Mlp Encoder part.
/// # Warning
/// The activation is applied at last layer, too.<br>
/// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output.
#[derive(Module, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
}

impl Mlp {
    /// create a new Mlp
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

    /// compute output
    /// # Warning
    /// the activation is applied last layer, too. <br>
    /// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output./// That is, linear -> activation -> linear -> activation -> ... -> linear -> activation = output.
    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let mut x = input;
        for layer in self.layers.iter() {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}

/// DiscreteQNet implementation with Mlp
#[derive(Module, Debug)]
pub struct MlpDiscreteQNet {
    encoder: Mlp,
    head: Linear,
}

impl MlpDiscreteQNet {
    /// create a new `MlpDiscreteQNet`
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpDiscreteQNet requires at least two dims: input dimension and the number of actions."); }
        let encoder = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let head = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        Self {
            encoder,
            head,
        }
    }
}

impl DiscreteQNet for MlpDiscreteQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        let x = self.encoder.forward(obs);
        self.head.forward(x)
    }
}

/// DiscreteDuelingQNet implementation with Mlp
#[derive(Module, Debug)]
pub struct MlpDiscreteDuelingQNet {
    encoder: Mlp,
    advantage_layer: Linear,
    value_layer: Linear,
}

impl MlpDiscreteDuelingQNet {
    /// create a new MlpDiscreteDuelingQNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpDiscreteDuelingQNet requires at least two dims: input dimension and the number of actions."); }
        let encoder = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let advantage_layer = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        let value_layer = LinearConfig::new(dims[dims.len() - 2], 1).init(device);
        
        Self {
            encoder,
            advantage_layer,
            value_layer,
        }
    }
}

impl DiscreteDuelingQNet for MlpDiscreteDuelingQNet {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>) {
        let x = self.encoder.forward(obs);
        let value = self.value_layer.forward(x.clone()).squeeze_dim(1);
        let advantage = self.advantage_layer.forward(x);
        (value, advantage)
    }
}

/// PolicyNet implementation with mlp
#[derive(Module, Debug)]
pub struct MlpPolicyNet {
    encoder: Mlp,
    head: Linear,
}

impl MlpPolicyNet {
    /// create a new MlpPolicy struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpPolicyNet requires at least two dims: input dimension and output dimension."); }
        let encoder = Mlp::new(&dims[..dims.len() - 1], activation, device);
        let head = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        
        Self {
            encoder,
            head
        }
    }
}

impl PolicyNet for MlpPolicyNet {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        self.head.forward(self.encoder.forward(obs))
    }
}

/// Separated ActorCritic implementation with mlp
#[derive(Module, Debug)]
pub struct MlpSeparatedActorCriticNet {
    actor_encoder: Mlp,
    critic_encoder: Mlp,
    actor_head: Linear,
    critic_head: Linear,
}

impl MlpSeparatedActorCriticNet {
    /// create a new MlpActorCriticNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpSeparatedActorCriticNet requires at least two dims: input dimension and output dimension."); }
        let actor_encoder = Mlp::new(&dims[..dims.len() - 1], activation.clone(), device);
        let critic_encoder = Mlp::new(&dims[..dims.len() - 1], activation, device);

        let actor_head = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        let critic_head = LinearConfig::new(dims[dims.len() - 2], 1).init(device);
        
        Self {
            actor_encoder,
            critic_encoder,
            actor_head,
            critic_head,
        }
    }
}

impl ActorCriticNet for MlpSeparatedActorCriticNet {
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

/// Shared ActorCritic implementation with mlp
#[derive(Module, Debug)]
pub struct MlpSharedActorCriticNet {
    encoder: Mlp,
    actor_head: Linear,
    critic_head: Linear,
}

impl MlpSharedActorCriticNet {
    /// create a new MlpSharedActorCriticNet struct with given dimensions and activation unit
    pub fn new(dims: &[usize], activation: ActivationConfig, device: &Device) -> Self {
        if dims.len() < 2 { panic!("MlpSharedActorCriticNet requires at least two dims: input dimension and output dimension."); }
        let encoder = Mlp::new(&dims[..dims.len() - 1], activation, device);

        let actor_head = LinearConfig::new(dims[dims.len() - 2], dims[dims.len() - 1]).init(device);
        let critic_head = LinearConfig::new(dims[dims.len() - 2], 1).init(device);
        
        Self {
            encoder,
            actor_head,
            critic_head,
        }
    }
}

impl ActorCriticNet for MlpSharedActorCriticNet {
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