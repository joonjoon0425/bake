//! A QNetwork trait for value-based methods, and basic helper for creating a new QNetwork
use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}};
use crate::{encoder::Encoder, head::Head, types::{ActionMask, Batchable}};

/// A QNetwork trait for value-based methods
pub trait QNetwork : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, obs: Self::Obs, mask: M) -> Tensor<2>;
}

/// A helper for creating encoder-head q network
#[derive(Module, Debug)]
pub struct SequentialQNetwork<E: Encoder, H: Head<Output = Tensor<2>>> {
    encoder: E,
    head: H,
}

impl<E: Encoder, H: Head<Output = Tensor<2>>> SequentialQNetwork<E, H> {
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head
        }
    }
}

impl<E: Encoder<Obs = Tensor<2>>, H: Head<Output = Tensor<2>>> QNetwork for SequentialQNetwork<E, H> {
    type Obs = Tensor<2>;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, obs: Self::Obs, mask: M) -> Tensor<2> {
        let qvalues = self.head.forward(self.encoder.forward(obs), mask, -1e9);
        qvalues
    }
}