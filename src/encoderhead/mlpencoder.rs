use std::marker::PhantomData;
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig, activation::Activation}, tensor::backend::Backend};
use crate::{encoderhead::Encoder, traits::Batchable};


#[derive(Module, Debug)]
pub struct MlpEncoder<B: Backend, Obs> {
    layers: Vec<Linear<B>>,
    activation: Activation<B>,
    #[module(skip)]
    _obs: PhantomData<Obs>
}

impl<B: Backend, Obs: Batchable<B, Batched = Tensor<B, D>>, const D: usize> MlpEncoder<B, Obs> {
    pub fn new(dims: Vec<usize>, activation: Activation<B>, device: &B::Device) -> Self {
        if dims.len() < 2 { panic!("MlpEncoder requires at least two dims: input dimension and output dimension."); }

        let mut layers = Vec::with_capacity(dims.len());
        
        for (i, &dim) in dims[..dims.len() - 1].iter().enumerate() {
            layers.push(LinearConfig::new(dim, dims[i + 1]).init(device))
        }
        
        Self {
            layers,
            activation,
            _obs: PhantomData
        }
    }
}

impl<B: Backend, Obs: Batchable<B, Batched = Tensor<B, D>>, const D: usize> Encoder<B, D> for MlpEncoder<B, Obs> {
    type Obs = Obs;

    fn forward(&self, batched_obs: <Self::Obs as Batchable<B>>::Batched) -> Tensor<B, D> {
        let mut x = batched_obs;
        for layer in &self.layers {
            x = self.activation.forward(layer.forward(x));
        }
        x
    }
}