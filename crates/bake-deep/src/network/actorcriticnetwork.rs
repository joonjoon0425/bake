//! Network structure for actor-critic method
//! 

use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}};

use crate::{distribution::Distribution, encoder::Encoder, head::{Head, VHead}, types::Batchable};

pub trait ActorCriticNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    type Dist: Distribution;
    type Constraint: Batchable;

    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist;
    fn critic(&self, obs: Self::Obs) -> Tensor<1>;

    /// For the encoder-sharing network, this function must be overloaded appropriately
    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> (Self::Dist, Tensor<1>) {
        (self.actor(obs.clone(), constraint), self.critic(obs))
    }
}

/// A helper for creating an ActorCriticNetwork
#[derive(Module, Debug)]
pub struct SequentialActorCriticNetwork<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    actor_encoder: E,
    critic_encoder: E,
    actor: H1,
    critic: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SequentialActorCriticNetwork<E, H1, H2> {
    pub fn new(actor_encoder: E, critic_encoder: E, actor: H1, critic: H2) -> Self {
        Self {
            actor_encoder,
            critic_encoder,
            actor,
            critic
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCriticNetwork for SequentialActorCriticNetwork<E, H1, H2> {
    type Obs = E::Obs;
    type Dist = H1::Output;
    type Constraint = H1::Constraint;

    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        self.actor.forward(self.actor_encoder.forward(obs), constraint)
    }

    fn critic(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic.forward(self.critic_encoder.forward(obs))
    }
}

/// A helper for creating an encoder-sharing actor-critic network
#[derive(Module, Debug)]
pub struct SharedActorCriticNetwork<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    encoder: E,
    actor: H1,
    critic: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SharedActorCriticNetwork<E, H1, H2> {
    pub fn new(encoder: E, actor: H1, critic: H2) -> Self {
        Self {
            encoder,
            actor,
            critic
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCriticNetwork for SharedActorCriticNetwork<E, H1, H2> {
    type Obs = E::Obs;
    type Dist = H1::Output;
    type Constraint = H1::Constraint;

    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        self.actor.forward(self.encoder.forward(obs), constraint)
    }

    fn critic(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic.forward(self.encoder.forward(obs))
    }

    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> (Self::Dist, Tensor<1>) {
        let encoded = self.encoder.forward(obs);
        let dist = self.actor.forward(encoded.clone(), constraint);
        let value = self.critic.forward(encoded);
        (dist, value)
    }
}