//! An A2C algorithm implementation
//! 
//! 
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::TensorData};

use crate::{distribution::Distribution, network::ActorCriticNetwork, types::Batch};
/// An Advantage Actor-Critic Algorithm implmentation
pub struct A2CAgent<Net: ActorCriticNetwork> {
    gamma: f32,
    c_e: f32,
    config: A2CConfig,
    net: Net,
}

impl<Net: ActorCriticNetwork> A2CAgent<Net> {
    /// create a new A2CAgent
    pub fn new(gamma: f32, c_e: f32, config: A2CConfig, net: Net) -> Self {
        Self {
            gamma,
            c_e,
            config,
            net,
        }
    }
    /// sample an action
    pub fn action(&self, obs: Net::Obs, constraint: Net::Constraint) -> <Net::Dist as Distribution>::Action {
        let (dist, _) = self.net.forward(obs, constraint);
        dist.sample()
    }

    /// get the most-likely action
    pub fn mode(&self, obs: Net::Obs, constraint: Net::Constraint) -> <Net::Dist as Distribution>::Action {
        let (dist, _) = self.net.forward(obs, constraint);
        dist.mode()
    }

    /// update the value network and policy
    pub fn update(mut self, lambda: f32, t: Batch<Net::Obs, <Net::Dist as Distribution>::Action, Net::Constraint>) -> (Self, Tensor<1>, Tensor<1>) {
        let n = t.batch_size;
        let (dist, values) = self.net.forward(t.obss, t.constraints);
        let (_, next_values) = self.net.forward(t.next_obss, t.next_constraints);
        // note that in the obss there are no terminal states & truncated states, since env's reset is called whenever next_obss as terminal states.
        let deltas = t.rewards + self.gamma * next_values * (1f32 - t.terminated.clone()) - values.clone();
        let deltas: Vec<f32> = deltas.into_data().into_vec().unwrap();
        let mut gae = vec![0f32; n];
        let terminated: Vec<f32> = t.terminated.into_data().into_vec().unwrap();
        let truncated: Vec<f32> = t.truncated.into_data().into_vec().unwrap();
        gae[n - 1] = deltas[n - 1];
        for i in (0..n-1).rev() {
            gae[i] = deltas[i] + self.gamma * lambda * gae[i + 1] * (1f32 - truncated[i]) * (1f32 - terminated[i]);
        }
        // 1. advantage
        let gae_raw = Tensor::from_data(TensorData::new(gae, [n]), &values.device());
        let gae = (gae_raw.clone() - gae_raw.clone().mean()) / (gae_raw.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(t.actions);
        let policy_loss = -(log_prob * gae).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let targets = gae_raw + values.clone().detach();
        let value_loss = MseLoss.forward(values, targets, Reduction::Mean);
        // 5. backward
        match self.config {
            A2CConfig::Shared{ lr, c_v, mut opt } => {
                let loss = policy_loss.clone() - entropy.clone() * self.c_e + value_loss.clone() * c_v;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt.step(lr, self.net, grads);
                self.config = A2CConfig::Shared { lr, c_v, opt };
            },
            A2CConfig::Separated { lr_p, mut opt_p, lr_v, mut opt_v } => {
                let loss = policy_loss.clone() - entropy.clone() * self.c_e;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt_p.step(lr_p, self.net, grads);

                let grads = value_loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt_v.step(lr_v, self.net, grads);
                self.config = A2CConfig::Separated { lr_p, opt_p, lr_v, opt_v };
            },
        }
        (self, value_loss, entropy)
    }
}

/// enum for branching between encoder-sharing and encoder-separated
pub enum A2CConfig {
    /// An encoder-sharing variant
    Shared{
        /// learning rate
        lr: f64,
        /// scales value loss
        c_v: f32,
        /// optimizer
        opt: ModuleOptimizer
    },
    /// An encoder-separated variant
    Separated{
        /// learning rate for policy net
        lr_p: f64,
        /// optimier for policy net
        opt_p: ModuleOptimizer, 
        /// learning rate for value net
        lr_v: f64,
        /// optimier for value net
        opt_v: ModuleOptimizer}
}

impl A2CConfig {
    /// A configuration for encoder-sharing actor-critic method
    pub fn shared(lr: f64, c_v: f32, opt: ModuleOptimizer) -> Self {
        Self::Shared {
            lr,
            c_v,
            opt,
        }
    }
    /// A configuration for encoder-separated actor-critic method
    pub fn separated(lr_p: f64, opt_p: ModuleOptimizer, lr_v: f64, opt_v: ModuleOptimizer) -> Self {
        Self::Separated {
            lr_p,
            opt_p,
            lr_v,
            opt_v,
        }
    }
}