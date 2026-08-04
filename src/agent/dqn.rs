use burn::{Tensor, module::AutodiffModule, nn::loss, optim::{GradientsParams, Optimizer}, tensor::{Bool, Int, backend::AutodiffBackend}};

use crate::{encoderhead::{AutodiffEncoder, AutodiffHead, EncoderHead, Head}, exploration::EpsGreedy, traits::Batchable, transition::BatchedTransition};

pub struct DqnAgent<B, E, H, O> //, const D1: usize, const D2: usize>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2, Obs: Batchable<B::InnerBackend>>,
    H: AutodiffHead<B, 2, Output = Tensor<B, 2>>,
    H::InnerModule: Head<B::InnerBackend, 2, Output = Tensor<B::InnerBackend, 2>>
{
    online: EncoderHead<B, E, H, 2>,
    target: EncoderHead<B, E, H, 2>,
    gamma: f32,
    optimizer: O,
    lr: f64,
    device: B::Device,
}

impl<B, E, H, O> DqnAgent<B, E, H, O>//, const D1: usize, const D2: usize> DqnAgent<B, E, H, O, D1, D2>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2>,
    H: AutodiffHead<B, 2, Output = Tensor<B, 2>>,
    H::InnerModule: Head<B::InnerBackend, 2, Output = Tensor<B::InnerBackend, 2>>,
    O: Optimizer<EncoderHead<B, E, H, 2>, B>,
{
    pub fn new(gamma: f32, encoder_head_network: EncoderHead<B, E, H, 2>, optimizer: O, lr: f64, device: B::Device) -> Self {
        let target = encoder_head_network.clone();
        DqnAgent {
            gamma,
            online: encoder_head_network,
            target,
            optimizer,
            lr,
            device
        }
    }

    pub fn update(mut self, transitions: BatchedTransition<B, <E::Obs as Batchable<B>>::Batched, Tensor<B, 1, Int>>) -> Self {
        let qvalues = self.online.forward(transitions.observations);
        let qvalues: Tensor<B, 1> = qvalues.gather(1, transitions.actions.unsqueeze_dim(1)).squeeze_dim(1);

        let target_q: Tensor<B, 1> = self.target.forward(transitions.next_observations).detach().max_dim(1).squeeze_dim(1);
        let target = transitions.rewards + self.gamma * target_q * (1f32 - transitions.terminated);
        
        let td_error = target.clone() - qvalues.clone();

        let loss = loss::MseLoss::new().forward(qvalues, target, loss::Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);
        self.online = self.optimizer.step(self.lr, self.online, grads);

        self
    }

    pub fn select_action(&self, exploration: &mut EpsGreedy, obs: E::Obs) -> i64 {
        let network = self.online.valid();
        let qvalues = network.forward_single(obs, &self.device);
        exploration.select_action(qvalues.squeeze_dim::<1>(0))
    }

    pub fn select_action_masked(&self, exploration: &mut EpsGreedy, obs: E::Obs, mask: Tensor<B::InnerBackend, 1, Bool>) -> i64 {
        let network = self.online.valid();
        let qvalues = network.forward_single(obs, &self.device);
        exploration.select_action_masked(qvalues.squeeze_dim::<1>(0), mask)
    }
}