use burn::{Tensor, nn::loss, optim::{GradientsParams, Optimizer}, tensor::{ElementConversion, TensorData, activation::{log_softmax, softmax}, backend::{AutodiffBackend, Backend}}};
use rand::{SeedableRng, distr::{Distribution, weighted::WeightedIndex}, rngs::StdRng};

use crate::{encoderhead::{self, Encoder, EncoderHead, Head}, traits::Batchable, transition::{BatchedTransition, Transition}};

// Advantage actor-critic algorithm with discrete action.
pub struct A2CAgent<B, E1, H1, E2, H2, O1, O2>
where 
    B: AutodiffBackend,
    E1: Encoder<B, 2>,
    H1: Head<B, 2, Output = Tensor<B, 2>>,
    E2: Encoder<B, 2, Obs = E1::Obs>,
    H2: Head<B, 2, Output = Tensor<B, 2>>,
{
    actor: EncoderHead<B, E1, H1, 2>,
    critic: EncoderHead<B, E2, H2, 2>,
    actor_optimizer: O1,
    critic_optimizer: O2,

    gamma: f32,
    entropy_coeff: f32,

    lr_a: f64,
    lr_c: f64,

    rng: StdRng,

    device: B::Device
}

impl<B, E1, H1, E2, H2, O1, O2> A2CAgent<B, E1, H1, E2, H2, O1, O2>
where 
    B: AutodiffBackend,
    E1: Encoder<B, 2>,
    H1: Head<B, 2, Output = Tensor<B, 2>>,
    E2: Encoder<B, 2, Obs = E1::Obs>,
    H2: Head<B, 2, Output = Tensor<B, 2>>,
    O1: Optimizer<EncoderHead<B, E1, H1, 2>, B>,
    O2: Optimizer<EncoderHead<B, E2, H2, 2>, B>
{
    pub fn new(seed: u64, gamma: f32, entropy_coeff: f32, lr_a: f64, lr_c: f64, device: B::Device, actor: EncoderHead<B, E1, H1, 2>, critic: EncoderHead<B, E2, H2, 2>, actor_optimizer: O1, critic_optimizer: O2) -> Self {
        Self {
            gamma,
            device,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            entropy_coeff,

            rng: StdRng::seed_from_u64(seed),

            lr_a,
            lr_c,
        }
    }

    pub fn select_action(&mut self, obs: E1::Obs) -> i64 {
        let logits: Tensor<B, 1> = self.actor.forward_single(obs, &self.device).detach().squeeze_dim(0);
        let probs: Vec<f32> = softmax(logits, 0).into_data().into_vec().unwrap();
        let dist = WeightedIndex::new(probs).unwrap();

        dist.sample(&mut self.rng) as i64
    }

    pub fn update(mut self, transitions: BatchedTransition<B, <E1::Obs as Batchable>::Batched<B>, <i64 as Batchable>::Batched<B>, ()>) 
    -> (Self, f32, f32) {
        let values: Tensor<B, 1> = self.critic.forward(transitions.observations.clone()).squeeze_dim(1);
        let next_values: Vec<f32> = self.critic.forward(transitions.next_observations.clone()).detach().into_data().into_vec().unwrap();

        let n = transitions.batch_size;
        let mut returns = vec![0f32; n + 1];
        let rewards: Vec<f32> = transitions.rewards.to_data().into_vec().unwrap();
        let terminated: Vec<f32> = transitions.terminated.to_data().into_vec().unwrap();
        let truncated: Vec<f32> = transitions.truncated.to_data().into_vec().unwrap();

        returns[n] = if terminated[n - 1] == 1f32 { 0f32 } else { next_values[n - 1] };

        for t in (0..n).rev() {
            returns[t] = if truncated[t] != 0.0 || terminated[t] != 0.0 {
                rewards[t] + self.gamma * next_values[t] * (1.0 - terminated[t])
            } else {
                rewards[t] + self.gamma * returns[t + 1]
            }
        }
        // 1. advantage
        returns.truncate(n);
        let returns: Tensor<B, 1> = Tensor::from_data(TensorData::new(returns, [n]), &self.device);
        let advantages = (returns.clone() - values.clone()).detach();
        let advantages = (advantages.clone() - advantages.clone().mean()) / (advantages.var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let logits = self.actor.forward(transitions.observations.clone());
        let log_probs = log_softmax(logits, 1);
        let selected = log_probs.clone().gather(1, transitions.actions.unsqueeze_dim(1)).squeeze_dim(1);
        let policy_loss = -(selected * advantages).mean();
        // 3. entropy
        let probs = log_probs.clone().exp();
        let entropy = -(probs * log_probs).sum_dim(1).mean();
        // 4. value loss
        let value_loss = loss::MseLoss.forward(values, returns, loss::Reduction::Mean);
        // 5. backward
        let actor_loss = policy_loss - entropy.clone() * self.entropy_coeff;
        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.actor);
        self.actor = self.actor_optimizer.step(self.lr_a, self.actor, grads);

        let grads = value_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.critic);
        self.critic = self.critic_optimizer.step(self.lr_c, self.critic, grads);

        (self, entropy.mean().detach().into_scalar().elem(), value_loss.into_scalar().elem())
    }
}