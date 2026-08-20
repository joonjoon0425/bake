//! An implementation of Proximal Policy Optimization algorithm
//! 
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::GradientsParams, tensor::{Int, TensorData}};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{config::ActorCriticConfig, distribution::Distribution, network::ActorCriticNetwork, types::{Batch, Indexable}};

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
    pub fn update(mut self, minibatch_size: usize, t: Batch<Net::Obs, <Net::Dist as Distribution>::Action, Net::Constraint, Tensor<1>>) -> (Self, Tensor<1>){
        let n = t.batch_size;
        let device = t.rewards.device();
        let (_, values) = self.net.forward(t.obss.clone(), t.constraints.clone());
        let (_, next_values) = self.net.forward(t.next_obss.clone(), t.next_constraints.clone());
        // note that in the obss there are no terminal states & truncated states, since env's reset is called whenever next_obss as terminal states.
        let deltas = t.rewards.clone() + self.gamma * next_values * (1f32 - t.terminated.clone()) - values.clone();
        let deltas: Vec<f32> = deltas.into_data().into_vec().unwrap();
        let mut gae = vec![0f32; n];
        let terminated: Vec<f32> = t.terminated.clone().into_data().into_vec().unwrap();
        let truncated: Vec<f32> = t.truncated.clone().into_data().into_vec().unwrap();
        gae[n - 1] = deltas[n - 1];
        for i in (0..n-1).rev() {
            gae[i] = deltas[i] + self.gamma * self.lambda * gae[i + 1] * (1f32 - truncated[i]) * (1f32 - terminated[i]);
        }
        // 1. advantage
        let gae_raw = Tensor::from_data(TensorData::new(gae, [n]), &values.device());
        let gae = (gae_raw.clone() - gae_raw.clone().mean()) / (gae_raw.clone().var(0) + 1e-9).sqrt();
        // 2. targets
        let targets = gae_raw.clone() + values.clone();
        // 3. K epoch minibatch rollout
        for k in 0..self.epoch {
            let mut perm: Vec<i64> = (0..t.batch_size as i64).collect();
            perm.shuffle(&mut self.rng);
            for chunk in perm.chunks(minibatch_size) {
                let idx = Tensor::<1, Int>::from_data(TensorData::new(chunk.to_vec(), [chunk.len()]), &device);
                // 3-1. clipped policy surrogate
                let minibatch = t.select(idx.clone());
                let (dist, values) = self.net.forward(minibatch.obss, minibatch.constraints);
                let log_probs = dist.log_probs(minibatch.actions);
                let selected_gae = gae.clone().select(0, idx.clone());
                let selected_targets = targets.clone().select(0, idx);
                let ratio = (log_probs - minibatch.extras).exp();
                let stacked = Tensor::stack::<2>(vec![ratio.clone() * selected_gae.clone(), ratio.clamp(1f32 - self.eps, 1f32 + self.eps) * selected_gae.clone()], 1);
                let surrogate_loss = -stacked.min_dim(1).mean();
                // 3-2. value loss
                let value_loss = MseLoss::new().forward(values.clone(), selected_targets.clone().detach(), Reduction::Mean);
                // 3-3 entropy
                let entropy = dist.entropy().mean();

                // 3-4. backward
                match self.config {
                    ActorCriticConfig::Shared{ lr, c_v, mut opt } => {
                        let loss = surrogate_loss.clone() - entropy.clone() * self.c_e + value_loss.clone() * c_v;
                        let grads = loss.backward();
                        let grads = GradientsParams::from_grads(grads, &self.net);
                        self.net = opt.step(lr, self.net, grads);
                        self.config = ActorCriticConfig::Shared { lr, c_v, opt };
                    },
                    ActorCriticConfig::Separated { lr_p, mut opt_p, lr_v, mut opt_v } => {
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
        let (dist, _) = self.net.forward(t.obss, t.constraints);

        (self, dist.entropy().mean())
    }
}
