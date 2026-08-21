//! An implementation of Proximal Policy Optimization algorithm
//! 
use bake_macros::Batchable;
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::GradientsParams, tensor::{Int, TensorData}};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{config::ActorCriticConfig, distribution::Distribution, network::ActorCriticNetwork, types::{Batch, Batchable}, utils::gae};

/// An implementation of Proximal Policy Optimization algorithm
pub struct PPOAgent<Net: ActorCriticNetwork> {
    gamma: f32,
    lambda: f32,
    eps: f32,
    epoch: usize,
    net: Net,
    rng: StdRng,
    c_e: f32,
    config: ActorCriticConfig,
}

/// helper struct for PPO update
#[derive(Batchable, Debug, Clone)]
struct PPOExtra {
    old_log_probs: Tensor<1>,
    gae: Tensor<1>,
    targets: Tensor<1>,
}


impl<Net: ActorCriticNetwork> PPOAgent<Net> {
    /// create a new PPOAgent
    pub fn new(seed: u64, gamma: f32, lambda: f32, eps: f32, c_e: f32, epoch: usize, config: ActorCriticConfig, net: Net) -> Self {
        Self {
            gamma,
            lambda,
            eps,
            c_e,
            rng: StdRng::seed_from_u64(seed),
            epoch,
            config,
            net
        }
    }
    /// sample an action
    pub fn action(&self, obs: Net::Obs, constraint: Net::Constraint) -> <Net::Dist as Distribution>::Action {
        let (dist, _) = self.net.forward(obs, constraint);
        dist.sample()
    }

    /// get current log_prob distribution of actions
    pub fn dist(&self, obs: Net::Obs, constraint: Net::Constraint) -> Net::Dist {
        let (dist, _) = self.net.forward(obs, constraint);
        dist
    }

    /// update the approximators
    pub fn update(mut self, minibatch_size: usize, mut batch: Batch<Net::Obs, <Net::Dist as Distribution>::Action, Net::Constraint, Tensor<1>>) -> (Self, PPOLog){
        let device = batch.rewards.device();
        let (_, values) = self.net.forward(batch.obss.clone(), batch.constraints.clone());
        let (_, next_values) = self.net.forward(batch.next_obss.clone(), batch.next_constraints.clone());
        let (adv, ret) = gae(batch.rewards.clone(), values, next_values, batch.terminated.clone(), batch.truncated.clone(), self.gamma, self.lambda);
        // 1. advantage
        let gae = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. K epoch minibatch rollout
        let old_log_probs = std::mem::replace(&mut batch.extras, Tensor::zeros([1], &device));
        let batch = batch.add_extra(PPOExtra { old_log_probs: old_log_probs.clone(), gae, targets: ret});
        for _ in 0..self.epoch {
            let mut perm: Vec<i64> = (0..batch.batch_size() as i64).collect();
            perm.shuffle(&mut self.rng);
            for chunk in perm.chunks(minibatch_size) {
                let idx = Tensor::<1, Int>::from_data(TensorData::new(chunk.to_vec(), [chunk.len()]), &device);
                // 2-1. clipped policy surrogate
                let minibatch = batch.clone().select(idx);
                let (dist, values) = self.net.forward(minibatch.obss, minibatch.constraints);
                let log_probs = dist.log_probs(minibatch.actions);
                let ratio = (log_probs - minibatch.extras.old_log_probs).exp();
                let stacked = Tensor::stack::<2>(vec![ratio.clone() * minibatch.extras.gae.clone(), ratio.clamp(1f32 - self.eps, 1f32 + self.eps) * minibatch.extras.gae ], 1);
                let surrogate_loss = -stacked.min_dim(1).mean();
                // 2-2. value loss
                let value_loss = MseLoss::new().forward(values.clone(), minibatch.extras.targets.clone().detach(), Reduction::Mean);
                // 2-3 entropy
                let entropy = dist.entropy().mean();

                // 2-4. backward
                match self.config {
                    ActorCriticConfig::Shared{ lr, c_v, mut opt } => {
                        let loss = surrogate_loss.clone() - entropy.clone() * self.c_e + value_loss.clone() * c_v;
                        let grads = loss.backward();
                        let grads = GradientsParams::from_grads(grads, &self.net);
                        self.net = opt.step(lr, self.net, grads);
                        self.config = ActorCriticConfig::Shared { lr, c_v, opt };
                    },
                    ActorCriticConfig::Separated{ lr_p, mut opt_p, lr_v, mut opt_v } => {
                        let loss = surrogate_loss.clone() - entropy.clone() * self.c_e;
                        let grads = loss.backward();
                        let grads = GradientsParams::from_grads(grads, &self.net);
                        self.net = opt_p.step(lr_p, self.net, grads);

                        let grads = value_loss.backward();
                        let grads = GradientsParams::from_grads(grads, &self.net);
                        self.net = opt_v.step(lr_v, self.net, grads);
                        self.config = ActorCriticConfig::Separated { lr_p, opt_p, lr_v, opt_v };
                    },
                }
            }
        }
        let (dist, _) = self.net.forward(batch.obss, batch.constraints);
        let log_ratio = dist.log_probs(batch.actions) - old_log_probs;
        let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean();
        let clip_ratio = (log_ratio.exp() - 1f32).abs().greater_elem(self.eps).float().mean();
        let entropy = dist.entropy().mean();
        (self, PPOLog::new(entropy, approx_kl, clip_ratio))
    }
}

/// logging struct for PPO
#[derive(Debug, Clone, Default)]
pub struct PPOLog {
    /// entropy
    pub entropy: Option<Tensor<1>>,
    /// approximate KL divergence
    pub approx_kl: Option<Tensor<1>>,
    /// clipped ratio
    pub clip_ratio: Option<Tensor<1>>,
}

impl PPOLog {
    /// create a new PPOLog struct
    pub fn new(entropy: Tensor<1>, approx_kl: Tensor<1>, clip_ratio: Tensor<1>) -> Self {
        Self {
            entropy: entropy.into(),
            approx_kl: approx_kl.into(),
            clip_ratio: clip_ratio.into()
        }
    }
    /// entropy
    pub fn entropy(&self) -> f32 { self.entropy.clone().map_or(0f32, |q| q.into_scalar()) }
    /// approximate KL divergence
    pub fn approx_kl(&self) -> f32 { self.approx_kl.clone().map_or(0f32, |q| q.into_scalar()) }
    /// clipped ratio
    pub fn clip_ratio(&self) -> f32 { self.clip_ratio.clone().map_or(0f32, |q| q.into_scalar()) }
}