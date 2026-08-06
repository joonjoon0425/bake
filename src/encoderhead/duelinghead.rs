use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::backend::{AutodiffBackend, Backend}};

use crate::encoderhead::Head;

// Dueling Head for Deuling DQN, with mean baseline.
#[derive(Module, Debug)]
pub struct DeulingHead<B: Backend> {
    value: Linear<B>, // latent -> 1
    advantage: Linear<B>, // latent -> n_actions
}

impl<B: AutodiffBackend> DeulingHead<B> {
    pub fn new(d_latent: usize, d_advantage: usize, device: &B::Device) -> Self {
        Self {
            value: LinearConfig::new(d_latent, 1).init(device),
            advantage: LinearConfig::new(d_latent, d_advantage).init(device),
        }
    }
}

impl<B: AutodiffBackend> Head<B, 2> for DeulingHead<B> {
    type Output = Tensor<B, 2>;

    fn forward(&self, encoded: Tensor<B, 2>) -> Self::Output {
        let value = self.value.forward(encoded.clone());
        let advantage = self.advantage.forward(encoded);
        let a_mean = advantage.clone().mean_dim(1);
        value + (advantage - a_mean)
    }

}