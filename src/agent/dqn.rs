use burn::{Tensor, module::AutodiffModule, optim::Optimizer, tensor::{Bool, backend::AutodiffBackend}};

use crate::{encoderhead::{AutodiffEncoder, AutodiffHead, EncoderHead, Head}, exploration::EpsGreedy};

pub struct DqnAgent<B, E, H, O> //, const D1: usize, const D2: usize>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2>,
    H: AutodiffHead<B, 2, Output = Tensor<B, 2>>,
    H::InnerModule: Head<B::InnerBackend, 2, Output = Tensor<B::InnerBackend, 2>>
{
    online: EncoderHead<B, E, H, 2>,
    target: EncoderHead<B, E, H, 2>,
    gamma: f32,
    optimizer: O
}

// Optimizer<EncoderHeadNetwork<B, E, H, D2>, B>

impl<B, E, H, O> DqnAgent<B, E, H, O>//, const D1: usize, const D2: usize> DqnAgent<B, E, H, O, D1, D2>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2>,
    H: AutodiffHead<B, 2, Output = Tensor<B, 2>>,
    H::InnerModule: Head<B::InnerBackend, 2, Output = Tensor<B::InnerBackend, 2>>,
    O: Optimizer<EncoderHead<B, E, H, 2>, B>,
{
    pub fn new(gamma: f32, encoder_head_network: EncoderHead<B, E, H, 2>, optimizer: O) -> Self {
        let target = encoder_head_network.clone();
        DqnAgent {
            gamma,
            online: encoder_head_network,
            target,
            optimizer
        }
    }

    pub fn update() {
        todo!()
    }

    pub fn select_action(&self, exploration: &EpsGreedy, obs: E::Obs) -> i64 {
        let network = self.online.valid();
        let qvalues = network.forward(obs);
        exploration.select_action(qvalues.squeeze_dim::<1>(0))
    }

    pub fn select_action_masked(&self, exploration: &EpsGreedy, obs: E::Obs, mask: Tensor<B::InnerBackend, 1, Bool>) -> i64 {
        let network = self.online.valid();
        let qvalues = network.forward(obs);
        exploration.select_action_masked(qvalues.squeeze_dim::<1>(0), mask)
    }
}