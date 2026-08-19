//! Network structure for actor-critic method
//! 

use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}};

use crate::{distribution::Distribution, encoder::Encoder, head::{Head, VHead}, types::Batchable};

pub trait ActorCriticNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    type Dist: Distribution;
    type Barrier: Batchable;

    fn forward(&self, obs: Self::Obs, barrier: Option<Self::Barrier>) -> (Self::Dist, Tensor<1>);
}

/// A helper for creating an ActorCriticNetwork
#[derive(Module, Debug)]
pub struct SequentialActorCriticNetwork<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    policy_encoder: E,
    value_encoder: E,
    policy: H1,
    value: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SequentialActorCriticNetwork<E, H1, H2> {
    pub fn new(policy_encoder: E, value_encoder: E, policy: H1, value: H2) -> Self {
        Self {
            policy_encoder,
            value_encoder,
            policy,
            value
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCriticNetwork for SequentialActorCriticNetwork<E, H1, H2> {
    type Obs = E::Obs;
    type Dist = H1::Output;
    type Barrier = H1::Barrier;

    fn forward(&self, obs: Self::Obs, barrier: Option<H1::Barrier>) -> (Self::Dist, Tensor<1>) {
        let dist = self.policy.forward(self.policy_encoder.forward(obs.clone()), barrier);
        let value = self.value.forward(self.value_encoder.forward(obs));
        (dist, value)
    }
}

/// A helper for creating an encoder-sharing actor-critic network
#[derive(Module, Debug)]
pub struct SharedActorCriticNetwork<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> {
    encoder: E,
    policy: H1,
    value: H2,
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> SharedActorCriticNetwork<E, H1, H2> {
    pub fn new(encoder: E, policy: H1, value: H2) -> Self {
        Self {
            encoder,
            policy,
            value
        }
    }
}

impl<E: Encoder, H1: Head<Output: Distribution>, H2: VHead> ActorCriticNetwork for SharedActorCriticNetwork<E, H1, H2> {
    type Obs = E::Obs;
    type Dist = H1::Output;
    type Barrier = H1::Barrier;

    fn forward(&self, obs: Self::Obs, barrier: Option<Self::Barrier>) -> (Self::Dist, Tensor<1>) {
        let encoded = self.encoder.forward(obs);
        let dist = self.policy.forward(encoded.clone(), barrier);
        let value = self.value.forward(encoded);
        (dist, value)
    }
}