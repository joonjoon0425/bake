//! A QNetwork trait for value-based methods, and basic helper for creating a new QNetwork
use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}};
use crate::{encoder::Encoder, head::QHead, types::{Batchable, DiscreteConstraint}};

/// A QNetwork trait for value-based methods
pub trait QNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;

    fn forward(&self, obs: Self::Obs, barrier: impl DiscreteConstraint) -> Tensor<2>;
}

/// A helper for creating encoder-head q network
#[derive(Module, Debug)]
pub struct SequentialQNetwork<E: Encoder, H: QHead> {
    encoder: E,
    head: H,
}

impl<E: Encoder, H: QHead> SequentialQNetwork<E, H> {
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head
        }
    }
}

impl<E: Encoder<Obs = Tensor<2>>, H: QHead> QNetwork for SequentialQNetwork<E, H> {
    type Obs = Tensor<2>;

    fn forward(&self, obs: Self::Obs, barrier: impl DiscreteConstraint) -> Tensor<2> {
        let qvalues = self.head.forward(self.encoder.forward(obs), barrier);
        qvalues
    }
}