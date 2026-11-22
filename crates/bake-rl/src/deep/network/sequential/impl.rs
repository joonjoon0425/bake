use burn::nn::*;
use burn::prelude::*;
use crate::deep::network::layer::noisy_linear::NoisyLinear;

use super::ForwardModule;

impl<const D: usize> ForwardModule<Tensor<D>, Tensor<D>> for Linear {
    fn forward(&self, input: Tensor<D>) -> Tensor<D> {
        Linear::forward(self, input)
    }
}

impl<const D: usize> ForwardModule<Tensor<D>, Tensor<D>> for NoisyLinear {
    fn forward(&self, input: Tensor<D>) -> Tensor<D> {
        NoisyLinear::forward(self, input)
    }
}

impl<const D: usize> ForwardModule<Tensor<D>, Tensor<D>> for Relu {
    fn forward(&self, input: Tensor<D>) -> Tensor<D> {
        Relu::forward(self, input)
    }
}