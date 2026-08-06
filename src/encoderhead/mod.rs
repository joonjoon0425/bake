use std::marker::PhantomData;
use burn::{module::{AutodiffModule, ModuleDisplay}, prelude::*, tensor::backend::AutodiffBackend};
use crate::traits::Batchable;

// Encoder trait. Converts the observation to Tensor
pub trait Encoder<B: AutodiffBackend, const BATCHED_RANK: usize> : AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay {
    type Obs: Batchable;

    fn forward(&self, batched_obs: <Self::Obs as Batchable>::Batched<B>) -> Tensor<B, BATCHED_RANK>;
    // this function will automatically produce a batched obs and produce input.
    fn forward_single(&self, obs: Self::Obs, device: &B::Device) -> Tensor<B, BATCHED_RANK> {
        self.forward(Self::Obs::batch(vec![obs], device))
    }
}

// Head trait. Take the encoded observation and make desired output
pub trait Head<B: AutodiffBackend, const BATCHED_RANK: usize> : AutodiffModule<B, InnerModule: ModuleDisplay> + ModuleDisplay {
    type Output;

    fn forward(&self, encoded: Tensor<B, BATCHED_RANK>) -> Self::Output;
}

// // alias trait for Autodiff Encoder
// pub trait AutodiffEncoder<B: AutodiffBackend, const D: usize> : Encoder<B, D, Obs: Batchable> + AutodiffModule<B, InnerModule: Encoder<B::InnerBackend, D, Obs = Self::Obs>>
// {}

// // blanket implementation for Encoders created with Module derive macro
// impl<B, E, const D: usize> AutodiffEncoder<B, D> for E
// where
//     B: AutodiffBackend,
//     E: Encoder<B, D, Obs: Batchable<B> + Batchable<B::InnerBackend>> + AutodiffModule<B, InnerModule: Encoder<B::InnerBackend, D, Obs = Self::Obs>>,
//     E::InnerModule: ModuleDisplay,
// {}

// // alias trait for Autodiff Head
// pub trait AutodiffHead<B: AutodiffBackend, const D: usize> : Head<B, D> + AutodiffModule<B, InnerModule: Head<B::InnerBackend, D>>
// {}

// // blanket implementation for Heads created with Module derive macro
// impl<B, H, const D: usize> AutodiffHead<B, D> for H
// where
//     B: AutodiffBackend,
//     H: Head<B, D> + AutodiffModule<B, InnerModule: Head<B::InnerBackend, D>>
// {}


// The connection of encoder and head
#[derive(Module, Debug)]
pub struct EncoderHead<B: Backend, E, H, const D: usize> {
    encoder: E,
    head: H,
    _backend: PhantomData<B>
}
// If I make a trait bound for E and H, the Module macro shoots error. The solution is to not bound the trait in the structure, but to bound it on impl blocks.

impl<B: AutodiffBackend, E: Encoder<B, D>, H: Head<B, D>, const D: usize> EncoderHead<B, E, H, D> {
    pub fn new(encoder: E, head: H) -> Self {
        Self {
            encoder,
            head,
            _backend: PhantomData,
        }
    }
}

impl<B: AutodiffBackend, E, H, const D: usize> EncoderHead<B, E, H, D>
where
    E: Encoder<B, D>,
    H: Head<B, D>,
{
    pub fn forward_single(&self, obs: E::Obs, device: &B::Device) -> H::Output {
        self.head.forward(self.encoder.forward_single(obs, device))
    }

    pub fn forward(&self, batched_obs: <E::Obs as Batchable>::Batched<B>) -> H::Output {
        self.head.forward(self.encoder.forward(batched_obs))
    }
}

pub mod mlpencoder; pub use mlpencoder::*;
pub mod linearhead; pub use linearhead::*;
pub mod duelinghead; pub use duelinghead::*;