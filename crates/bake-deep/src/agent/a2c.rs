//! An A2C algorithm implementation
//! 
//! 
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::TensorData};

use crate::{config::ActorCriticConfig, distribution::Distribution, network::ActorCriticNetwork, types::Batch};
/// An Advantage Actor-Critic Algorithm implmentation
pub struct A2CAgent<Net: ActorCriticNetwork> {
    gamma: f32,
    lambda: f32,
    c_e: f32,
    config: ActorCriticConfig,
    net: Net,
}

impl<Net: ActorCriticNetwork> A2CAgent<Net> {
    /// create a new A2CAgent
    pub fn new(gamma: f32, lambda: f32, c_e: f32, config: ActorCriticConfig, net: Net) -> Self {
        Self {
            gamma,
            c_e,
            lambda,
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
    pub fn update(mut self, t: Batch<Net::Obs, <Net::Dist as Distribution>::Action, Net::Constraint>) -> (Self, Tensor<1>, Tensor<1>) {
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
            gae[i] = deltas[i] + self.gamma * self.lambda * gae[i + 1] * (1f32 - truncated[i]) * (1f32 - terminated[i]);
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
            ActorCriticConfig::Shared{ lr, c_v, mut opt } => {
                let loss = policy_loss.clone() - entropy.clone() * self.c_e + value_loss.clone() * c_v;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt.step(lr, self.net, grads);
                self.config = ActorCriticConfig::Shared { lr, c_v, opt };
            },
            ActorCriticConfig::Separated { lr_p, mut opt_p, lr_v, mut opt_v } => {
                let loss = policy_loss.clone() - entropy.clone() * self.c_e;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt_p.step(lr_p, self.net, grads);

                let grads = value_loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = opt_v.step(lr_v, self.net, grads);
                self.config = ActorCriticConfig::Separated { lr_p, opt_p, lr_v, opt_v };
            },
        }
        (self, value_loss, entropy)
    }
}