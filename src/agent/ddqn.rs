use burn::{Tensor, nn::loss, optim::{GradientsParams, Optimizer}, tensor::{ElementConversion, Int, backend::AutodiffBackend}};

use crate::{encoderhead::{Encoder, EncoderHead, Head, LinearHead}, exploration::EpsGreedy, traits::Batchable, transition::BatchedTransition};

// Double Deep Q-Learning algorithm
pub struct DoubleDqnAgent<B, E, H, O>
where
    B: AutodiffBackend,
    E: Encoder<B, 2>,
    H: Head<B, 2, Output = Tensor<B, 2>>,
{
    online: EncoderHead<B, E, H, 2>,
    target: EncoderHead<B, E, H, 2>,
    gamma: f32,
    optimizer: O,
    lr: f64,
    device: B::Device,
}

impl<B, E, H, O> DoubleDqnAgent<B, E, H, O>
where
    B: AutodiffBackend,
    E: Encoder<B, 2>,
    H: Head<B, 2, Output = Tensor<B, 2>>,
    O: Optimizer<EncoderHead<B, E, H, 2>, B>,
{
    pub fn new(gamma: f32, encoder_head: EncoderHead<B, E, H, 2>, optimizer: O, lr: f64, device: B::Device) -> Self {
        let target = encoder_head.clone();
        DoubleDqnAgent {
            gamma,
            online: encoder_head,
            target,
            optimizer,
            lr,
            device
        }
    }

    pub fn update(mut self, transitions: BatchedTransition<B, <E::Obs as Batchable>::Batched<B>, Tensor<B, 1, Int>>) -> (Self, f32, f32, f32) {
        let batched_qvalues = self.online.forward(transitions.observations);
        let qvalues: Tensor<B, 1> = batched_qvalues.gather(1, transitions.actions.unsqueeze_dim(1)).squeeze_dim(1);

        let next_actions = self.online.forward(transitions.next_observations.clone()).detach().argmax(1);
        let next_q: Tensor<B, 1> = self.target.forward(transitions.next_observations).detach().gather(1, next_actions).squeeze_dim(1);
        let target = transitions.rewards + self.gamma * next_q * (1f32 - transitions.terminated);
        
        let td_error = target.clone() - qvalues.clone();
        let q_mean = qvalues.clone().mean().into_scalar().elem();

        let loss = loss::MseLoss::new().forward(qvalues, target, loss::Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);
        self.online = self.optimizer.step(self.lr, self.online, grads);

        (self, loss.into_scalar().elem(), td_error.mean().into_scalar().elem(), q_mean)
    }

    pub fn select_action(&self, exploration: &mut EpsGreedy, obs: E::Obs) -> i64 {
        let qvalues = self.online.forward_single(obs, &self.device).detach();
        exploration.select_action(qvalues.squeeze_dim::<1>(0))
    }

    pub fn select_action_masked(&self, exploration: &mut EpsGreedy, obs: E::Obs, mask: &[bool]) -> i64 {
        let qvalues = self.online.forward_single(obs, &self.device).detach();
        exploration.select_action_masked(qvalues.squeeze_dim::<1>(0), mask)
    }

    pub fn sync(&mut self) { self.target = self.online.clone() }
}