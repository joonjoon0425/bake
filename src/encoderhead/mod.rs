use std::marker::PhantomData;
use burn::{module::{AutodiffModule, ModuleDisplay}, prelude::*, tensor::backend::AutodiffBackend};

// Encoder trait. Converts the observation to Tensor
pub trait Encoder<B: Backend, const D: usize> : Module<B> + ModuleDisplay {
    type Obs;

    fn forward(&self, obs: Self::Obs) -> Tensor<B, D>;
}

// Head trait. Take the encoded observation and make desired output
pub trait Head<B: Backend, const D: usize> : Module<B> + ModuleDisplay {
    type Output;

    fn forward(&self, encoded: Tensor<B, D>) -> Self::Output;
}

// alias trait for Autodiff Encoder
pub trait AutodiffEncoder<B: AutodiffBackend, const D: usize> : Encoder<B, D> + AutodiffModule<B, InnerModule: Encoder<B::InnerBackend, D, Obs = Self::Obs>>
{}

// blanket implementation for Encoders created with Module derive macro
impl<B, E, const D: usize> AutodiffEncoder<B, D> for E
where
    B: AutodiffBackend,
    E: Encoder<B, D> + AutodiffModule<B, InnerModule: Encoder<B::InnerBackend, D, Obs = Self::Obs>>,
    E::InnerModule: ModuleDisplay,
{}

// alias trait for Autodiff Head
pub trait AutodiffHead<B: AutodiffBackend, const D: usize> : Head<B, D> + AutodiffModule<B, InnerModule: Head<B::InnerBackend, D>>
{}

// blanket implementation for Heads created with Module derive macro
impl<B, H, const D: usize> AutodiffHead<B, D> for H
where
    B: AutodiffBackend,
    H: Head<B, D> + AutodiffModule<B, InnerModule: Head<B::InnerBackend, D>>
{}


// The connection of encoder and head
#[derive(Module, Debug)]
pub struct EncoderHead<B: Backend, E, H, const D: usize> {
    encoder: E,
    head: H,
    backend: PhantomData<B>
}
// If I make a trait bound for E and H, the Module macro shoots error. The solution is to not bound the trait in the structure, but to bound it on impl blocks.

impl<B: Backend, E, H, const D: usize> EncoderHead<B, E, H, D>
where
    E: Encoder<B, D>,
    H: Head<B, D>,
{
    pub fn forward(&self, obs: E::Obs) -> H::Output {
        self.head.forward(self.encoder.forward(obs))
    }
}

pub mod identityencoder;
pub use identityencoder::*;