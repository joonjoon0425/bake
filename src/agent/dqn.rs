use burn::{Tensor, module::AutodiffModule, nn::loss, optim::{GradientsParams, Optimizer}, tensor::{Bool, ElementConversion, Int, backend::AutodiffBackend}};

use crate::{encoderhead::{AutodiffEncoder, EncoderHead, LinearHead}, exploration::EpsGreedy, traits::Batchable, transition::BatchedTransition};

pub struct DqnAgent<B, E, O>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2, Obs: Batchable<B::InnerBackend>>,
{
    online: EncoderHead<B, E, LinearHead<B>, 2>,
    target: EncoderHead<B, E, LinearHead<B>, 2>,
    gamma: f32,
    optimizer: O,
    lr: f64,
    device: B::Device,
}

impl<B, E, O> DqnAgent<B, E, O>
where
    B: AutodiffBackend,
    E: AutodiffEncoder<B, 2>,
    O: Optimizer<EncoderHead<B, E, LinearHead<B>, 2>, B>,
{
    pub fn new(gamma: f32, encoder_head_network: EncoderHead<B, E, LinearHead<B>, 2>, optimizer: O, lr: f64, device: B::Device) -> Self {
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

    pub fn update(mut self, transitions: BatchedTransition<B, <E::Obs as Batchable<B>>::Batched, Tensor<B, 1, Int>>) -> (Self, f32, f32, f32) {
        let qvalues = self.online.forward(transitions.observations);
        let qvalues: Tensor<B, 1> = qvalues.gather(1, transitions.actions.unsqueeze_dim(1)).squeeze_dim(1);

        let target_q: Tensor<B, 1> = self.target.forward(transitions.next_observations).detach().max_dim(1).squeeze_dim(1);
        let target = transitions.rewards + self.gamma * target_q * (1f32 - transitions.terminated);
        
        let td_error = target.clone() - qvalues.clone();
        let q_mean = qvalues.clone().mean().into_scalar().elem();

        let loss = loss::MseLoss::new().forward(qvalues, target, loss::Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);
        self.online = self.optimizer.step(self.lr, self.online, grads);

        (self, loss.into_scalar().elem(), td_error.mean().into_scalar().elem(), q_mean)
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

    pub fn sync(&mut self) { self.target = self.online.clone() }
}