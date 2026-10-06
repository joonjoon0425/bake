//! basic Mlp network implementations
use bake_macros::{actor_critic, policy, qnet};
use burn::prelude::*;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, activation::{Activation, ActivationConfig}};

use crate::distribution::Categorical;

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

/// Mlp network for plain dqn methods
#[derive(Module, Debug)]
pub struct MlpDiscreteQNet {
    encoder: Mlp,
    head: Linear,
}

#[qnet]
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
    
    /// return the action values
    pub fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        let x = self.encoder.forward(obs);
        self.head.forward(x)
    }
}
/// Mlp network for dueling dqn methods
#[derive(Module, Debug)]
pub struct MlpDiscreteDuelingQNet {
    encoder: Mlp,
    advantage_layer: Linear,
    value_layer: Linear,
}

#[qnet(dueling)]
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
    /// return the (value, advantage) tuple
    pub fn forward(&self, obs: Tensor<2>) -> (Tensor<1>, Tensor<2>) {
        let x = self.encoder.forward(obs);
        let value = self.value_layer.forward(x.clone()).squeeze_dim(1);
        let advantage = self.advantage_layer.forward(x);
        (value, advantage)
    }
}

/// Mlp network for policy gradient methods with discrete actions
#[derive(Module, Debug)]
pub struct MlpPolicy {
    encoder: Mlp,
    head: Linear,
}

#[policy(distribution = Categorical)]
impl MlpPolicy {
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
    
    fn forward(&self, obs: Tensor<2>) -> Tensor<2> {
        self.head.forward(self.encoder.forward(obs))
    }
}

/// Mlp network for separated encoder actor critic methods with discrete actions
#[derive(Module, Debug)]
pub struct MlpSeparatedActorCritic {
    actor_encoder: Mlp,
    critic_encoder: Mlp,
    actor_head: Linear,
    critic_head: Linear,
}

#[actor_critic(distribution = Categorical)]
impl MlpSeparatedActorCritic {
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

    /// returns the logits for Categorical distribution
    pub fn actor(&self, obs: Tensor<2>) -> Tensor<2> {
        self.actor_head.forward(self.actor_encoder.forward(obs))
    }
    /// returns the state value of given observation
    pub fn critic(&self, obs: Tensor<2>) -> Tensor<1> {
        self.critic_head.forward(self.critic_encoder.forward(obs)).squeeze_dim(1)
    }
}

/// Mlp network for shared encoder actor critic methods with discrete actions
#[derive(Module, Debug)]
pub struct MlpSharedActorCritic {
    encoder: Mlp,
    actor_head: Linear,
    critic_head: Linear,
}

#[actor_critic(distribution = Categorical, encoder_shared)]
impl MlpSharedActorCritic {
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
    /// returns the logits for Categorical distribution
    pub fn actor(&self, obs: Tensor<2>) -> Tensor<2> {
        self.actor_head.forward(self.encoder.forward(obs))
    }
    /// returns the state value of given observation
    pub fn critic(&self, obs: Tensor<2>) -> Tensor<1> {
        self.critic_head.forward(self.encoder.forward(obs)).squeeze_dim(1)
    }
    /// returns the (logit, state value) tuple
    pub fn actor_critic(&self, obs: Tensor<2>) -> (Tensor<2>, Tensor<1>) {
        let encoded = self.encoder.forward(obs);
        (self.actor_head.forward(encoded.clone()), self.critic_head.forward(encoded).squeeze_dim(1))
    }
}
