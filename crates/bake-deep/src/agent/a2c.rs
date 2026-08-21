//! An A2C algorithm implementation
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::GradientsParams};
use crate::{config::ActorCriticConfig, distribution::Distribution, network::ActorCriticNetwork, types::Batch, utils::gae};
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
    pub fn update(mut self, t: Batch<Net::Obs, <Net::Dist as Distribution>::Action, Net::Constraint>) -> (Self, A2CLog) {
        let (dist, values) = self.net.forward(t.obss, t.constraints);
        let (_, next_values) = self.net.forward(t.next_obss, t.next_constraints);
        let (adv, ret) = gae(t.rewards, values.clone(), next_values, t.terminated, t.truncated, self.gamma, self.lambda);
        // 1. advantage
        let gae = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(t.actions);
        let policy_loss = -(log_prob * gae).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let value_loss = MseLoss.forward(values, ret.detach(), Reduction::Mean);
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
        (self, A2CLog::new(value_loss, entropy))
    }
}

/// A logging struct for A2C
#[derive(Debug, Clone, Default)]
pub struct A2CLog {
    /// a value loss
    pub value_loss: Option<Tensor<1>>,
    /// entropy
    pub entropy: Option<Tensor<1>>,
}

impl A2CLog {
    /// create empty A2CLog
    pub fn new(value_loss: Tensor<1>, entropy: Tensor<1>) -> Self {
        Self {
            value_loss: value_loss.into(),
            entropy: entropy.into()
        }
    }
    /// get value loss in scalar
    pub fn value_loss(&self) -> f32 {
        self.value_loss.clone().map_or(0f32, |e| e.into_scalar())
    }
    /// get entropy in scalar
    pub fn entropy(&self) -> f32 {
        self.entropy.clone().map_or(0f32, |e| e.into_scalar())
    }
}