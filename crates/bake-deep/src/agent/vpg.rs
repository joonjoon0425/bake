//! Vanilla Policy Gradient (REINFORCE) algorithm implementation
use burn::{Tensor, optim::{GradientsParams, ModuleOptimizer}, tensor::TensorData};

use crate::{distribution::Distribution, network::LogitNetwork, types::{Batch, Batchable}};
/// A Vanilla Policy Gradient (REINFORCE) algorithm implementation
pub struct VPGAgent<LogitNet: LogitNetwork> {
    gamma: f32,
    baseline: Baseline,
    online: LogitNet,
    opt: ModuleOptimizer,
    c_e: f32,
    lr: f64,
}

impl<LogitNet: LogitNetwork> VPGAgent<LogitNet> {
    /// create a new VPGAgent
    pub fn new(gamma: f32, baseline: Baseline, net: LogitNet, c_e: f32, opt: ModuleOptimizer, lr: f64) -> Self {
        Self {
            gamma,
            baseline,
            online: net,
            opt,
            lr,
            c_e
        }
    }
    /// sample an action
    pub fn action(&self, obs: LogitNet::Obs, constraint: LogitNet::Constraint) -> <LogitNet::Dist as Distribution>::Action {
        let dist = self.online.forward(obs, constraint);
        dist.sample()
    }

    /// get the distribution
    pub fn mode(&self, obs: LogitNet::Obs, constraint: LogitNet::Constraint) -> LogitNet::Dist {
        let dist = self.online.forward(obs, constraint);
        dist
    }

    /// update the logit network
    pub fn update(mut self, t: Batch<LogitNet::Obs, <LogitNet::Dist as Distribution>::Action, LogitNet::Constraint>) -> (Self, VPGLog) {
        let len = t.batch_size();
        let device = t.device();
        let dist = self.online.forward(t.obss, t.constraints.into());
        let rewards = t.rewards.into_data().into_vec().unwrap();
        let mut returns = vec![0f32; len];
        returns[len - 1] = rewards[len - 1];
        for i in (0..(len - 1)).rev() {
            returns[i] = rewards[i] + self.gamma * returns[i + 1]
        }
        
        let returns = Tensor::from_data(TensorData::new(returns, [len]), &device);
        let returns = self.baseline.advantage(returns);

        let log_probs = dist.log_probs(t.actions);
        let entropy = dist.entropy();
        let loss = -(returns * log_probs).mean() - entropy.clone().mean() * self.c_e;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);

        self.online = self.opt.step(self.lr, self.online, grads);

        (self, VPGLog::new(loss, entropy.mean()))
    }
}

/// logging struct for VPG
#[derive(Debug, Clone, Default)]
pub struct VPGLog {
    /// surrogate loss
    pub surrogate_loss: Option<Tensor<1>>,
    /// entropy
    pub entropy: Option<Tensor<1>>,
}

impl VPGLog {
    /// create a new VPGLog struct
    pub fn new(surrogate_loss: Tensor<1>, entropy: Tensor<1>) -> Self {
        Self {
            surrogate_loss: surrogate_loss.into(),
            entropy: entropy.into()
        }
    }
    /// surrogate loss
    pub fn surrogate_loss(&self) -> f32 { self.surrogate_loss.clone().map_or(0f32, |q| q.into_scalar()) }
    /// entropy
    pub fn entropy(&self) -> f32 { self.entropy.clone().map_or(0f32, |q| q.into_scalar()) }
}

/// baseline enum for examining the effects of it
#[derive(Debug, Clone, Copy)]
pub enum Baseline {
    /// No baseline is applied
    None,
    /// Normalize the return
    Normalized,
    /// Set the mean of returns into zero
    Mean,
}

impl Baseline {
    /// calculate the return with baseline applied
    pub fn advantage(&self, returns: Tensor<1>) -> Tensor<1> {
        match self {
            Baseline::None => returns,
            Baseline::Normalized => {
                (returns.clone() - returns.clone().mean()) / (returns.var(0).sqrt() + 1e-8)
            }
            Baseline::Mean => {
                returns.clone() - returns.clone().mean()
            }
        }
    }
}