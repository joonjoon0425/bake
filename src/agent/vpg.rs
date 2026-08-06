use burn::{Tensor, optim::{GradientsParams, Optimizer}, tensor::{ElementConversion, Int, TensorData, activation::{log_softmax, softmax}, backend::{AutodiffBackend, Backend}}};
use rand::{SeedableRng, distr::{Distribution, weighted::WeightedIndex}, rngs::StdRng};
use crate::{encoderhead::{Encoder, EncoderHead, LinearHead}, traits::Batchable, transition::BatchedTransition};

// Vanilla Policy Gradient algorithm with discrete action.
pub struct VpgAgent<B, E, O>
where
    B: AutodiffBackend,
    E: Encoder<B, 2>
{
    gamma: f32,
    online: EncoderHead<B, E, LinearHead<B>, 2>,
    baseline: Baseline,
    optimizer: O,
    lr: f64,
    rng: StdRng,
    device: B::Device,
}

impl<B, E, O> VpgAgent<B, E, O>
where
    B: AutodiffBackend,
    E: Encoder<B, 2>,
    O: Optimizer<EncoderHead<B, E, LinearHead<B>, 2>, B>,
{
    pub fn new(seed: u64, gamma: f32, baseline: Baseline, encoder_head: EncoderHead<B, E, LinearHead<B>, 2>, optimizer: O, lr: f64, device: B::Device) -> Self {
        Self {
            gamma,
            baseline,
            online: encoder_head,
            optimizer,
            lr,
            device,
            rng: StdRng::seed_from_u64(seed)
        }
    }

    pub fn select_action(&self, obs: E::Obs) -> i64 {
        let logits: Tensor<B, 1> = self.online.forward_single(obs, &self.device).detach().squeeze_dim(0);
        let probs = softmax(logits, 0);
        let action = probs.categorical(1).into_scalar().elem();

        action
    }

    pub fn select_action_masked(&mut self, obs: E::Obs, mask: &[bool]) -> i64 {
        let logits: Tensor<B, 1> = self.online.forward_single(obs, &self.device).detach().squeeze_dim(0);
        let logits: Vec<f32> = logits.into_data().into_vec().unwrap();
        let (possible_actions, possible_logits): (Vec<usize>, Vec<f32>) = logits.iter().enumerate().filter(|(i, _)| mask[*i]).map(|(i, q)| (i, *q)).collect();

        let dist = WeightedIndex::new(possible_logits).unwrap();
        possible_actions[dist.sample(&mut self.rng)] as i64
    }

    pub fn update(mut self, episode: BatchedTransition<B, <E::Obs as Batchable>::Batched<B>, Tensor<B, 1, Int>>) -> (Self, f32) {
        let len = episode.batch_size;
        let rewards = episode.rewards.into_data().to_vec().unwrap();

        let mut returns = vec![0f32; len];
        returns[len - 1] = rewards[len - 1];
        for t in (0..(len - 1)).rev() {
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        }

        let returns = Tensor::from_data(TensorData::new(returns, [len]), &self.device);
        let returns = self.baseline.advantage(returns);

        let logits = self.online.forward(episode.observations);
        let log_probs = log_softmax(logits, 1);
        let selected_probs: Tensor<B, 1> = log_probs.gather(1, episode.actions.unsqueeze_dim(1)).squeeze_dim(1);
        
        let loss = -(returns * selected_probs).mean();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);

        self.online = self.optimizer.step(self.lr, self.online, grads);

        (self, loss.into_scalar().elem())
    }
}

// baseline enum for examining the effects of it
#[derive(Debug, Clone, Copy)]
pub enum Baseline {
    None,
    Normalized,
    Mean,
}

impl Baseline {
    pub fn advantage<B: Backend>(&self, returns: Tensor<B, 1>) -> Tensor<B, 1> {
        match self {
            Baseline::None => returns,
            Baseline::Normalized => {
                (returns.clone() - returns.clone().mean()) / returns.var(0).sqrt() + 1e-8
            }
            Baseline::Mean => {
                returns.clone() - returns.clone().mean()
            }
        }
    }
}