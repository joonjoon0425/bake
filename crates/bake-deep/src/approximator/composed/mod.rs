//! A helper structs for approximators

use burn::{Tensor, module::Module};
use crate::{approximator::*, constraint::DiscreteConstraint, distribution::Distribution, exploration::NoiseReset};

/// A helper for creating a LogitNetwork
#[derive(Module, Debug)]
pub struct ComposedPolicy<E: Encoder, H: Head<Output: Distribution>> {
    encoder: E,
    head: H
}

impl<E: Encoder, H: Head<Output: Distribution>> ComposedPolicy<E, H> {
    /// create a new composed policy
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head
        }
    }
}

impl<E: Encoder, H: Head<Output: Distribution>> Policy for ComposedPolicy<E, H> {
    type Obs = E::Obs;
    type Dist = H::Output;
    type Constraint = H::Constraint;

    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        self.head.forward(self.encoder.forward(obs), constraint)
    }
}

impl<E: Encoder + NoiseReset, H: Head<Output: Distribution> + NoiseReset> NoiseReset for ComposedPolicy<E, H> {
    fn reset_noise(&mut self) {
        self.encoder.reset_noise();
        self.head.reset_noise();
    }
}

/// A helper for creating encoder-head q network
#[derive(Module, Debug)]
pub struct ComposedQFunction<E: Encoder, H: QHead> {
    encoder: E,
    head: H,
}

impl<E: Encoder, H: QHead> ComposedQFunction<E, H> {
    /// create a new composed q function
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head
        }
    }
}

impl<E: Encoder, H: QHead> QFunction for ComposedQFunction<E, H> {
    type Obs = E::Obs;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let qvalues = self.head.forward(self.encoder.forward(obs), constraint);
        qvalues
    }
}

impl<E: Encoder + NoiseReset, H: QHead + NoiseReset> NoiseReset for ComposedQFunction<E, H> {
    fn reset_noise(&mut self) {
        self.encoder.reset_noise();
        self.head.reset_noise();
    }
}
/// A helper for creating an ActorCriticNetwork
#[derive(Module, Debug)]
pub struct SeparatedActorCritic<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    actor_encoder: E,
    critic_encoder: E,
    actor: H1,
    critic: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SeparatedActorCritic<E, H1, H2> {
    /// create a new encoder-separated actor and critic
    pub fn new(actor_encoder: E, critic_encoder: E, actor: H1, critic: H2) -> Self {
        Self {
            actor_encoder,
            critic_encoder,
            actor,
            critic
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCritic for SeparatedActorCritic<E, H1, H2> {
    type Obs = E::Obs;
    type Dist = H1::Output;
    type Constraint = H1::Constraint;

    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        self.actor.forward(self.actor_encoder.forward(obs), constraint)
    }

    fn critic(&self, obs: Self::Obs) -> Tensor<1> {
        self.critic.forward(self.critic_encoder.forward(obs))
    }

    fn shares_encoder(&self) -> bool { false }
}

impl<E: Encoder + NoiseReset, H1: Head<Output: Distribution> + NoiseReset, H2: VHead + NoiseReset> NoiseReset for SeparatedActorCritic<E, H1, H2> {
    fn reset_noise(&mut self) {
        self.critic_encoder.reset_noise();
        self.actor_encoder.reset_noise();
        self.critic.reset_noise();
        self.actor.reset_noise();
    }
}

/// A helper for creating an encoder-sharing actor-critic network
#[derive(Module, Debug)]
pub struct SharedActorCritic<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    encoder: E,
    actor: H1,
    critic: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SharedActorCritic<E, H1, H2> {
    /// creat a new encoder-shared actor and critic
    pub fn new(encoder: E, actor: H1, critic: H2) -> Self {
        Self {
            encoder,
            actor,
            critic
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCritic for SharedActorCritic<E, H1, H2> {
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

    fn shares_encoder(&self) -> bool { true }
}

impl<E: Encoder + NoiseReset, H1: Head<Output: Distribution> + NoiseReset, H2: VHead + NoiseReset> NoiseReset for SharedActorCritic<E, H1, H2> {
    fn reset_noise(&mut self) {
        self.encoder.reset_noise();
        self.critic.reset_noise();
        self.actor.reset_noise();
    }
}