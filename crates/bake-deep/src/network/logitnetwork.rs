//! A logits network for policy-based agents
use burn::module::{AutodiffModule, Module, ModuleDisplay};

use crate::{distribution::Distribution, encoder::Encoder, head::Head, types::Batchable};
pub trait LogitNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    type Dist: Distribution;
    type Barrier: Batchable;

    fn forward(&self, obs: Self::Obs, barrier: Self::Barrier) -> Self::Dist;
}

/// A helper for creating a LogitNetwork
#[derive(Module, Debug)]
pub struct SequentialLogitNetwork<E: Encoder, H: Head<Output: Distribution>> {
    encoder: E,
    head: H
}

impl<E: Encoder, H: Head<Output: Distribution>> SequentialLogitNetwork<E, H> {
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head
        }
    }
}

impl<E: Encoder, H: Head<Output: Distribution>> LogitNetwork for SequentialLogitNetwork<E, H> {
    type Obs = E::Obs;
    type Dist = H::Output;
    type Barrier = H::Barrier;

    fn forward(&self, obs: Self::Obs, constraint: Self::Barrier) -> Self::Dist {
        self.head.forward(self.encoder.forward(obs), constraint)
    }
}