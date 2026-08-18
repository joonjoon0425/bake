//! An A2C algorithm implementation
//! 
//! 
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::TensorData};

use crate::{distribution::Distribution, network::ActorCriticNetwork, types::{ActionMask, Batch}};
/// An Advantage Actor-Critic Algorithm implmentation
pub struct A2CAgent<Net: ActorCriticNetwork> {
    gamma: f32,
    net: Net,
    c_v: f32,
    c_e: f32,
    lr: f64,
    opt: ModuleOptimizer,
}

impl<Net: ActorCriticNetwork> A2CAgent<Net> {
    /// create a new A2CAgent
    pub fn new(gamma: f32, c_v: f32, c_e: f32, lr: f64, opt: ModuleOptimizer, net: Net) -> Self {
        Self {
            gamma,
            c_v,
            c_e,
            lr,
            opt,
            net,
        }
    }
    /// sample an action
    pub fn action<M: ActionMask<Value = Tensor<2>>>(&self, obs: Net::Obs, mask: M) -> <Net::Dist as Distribution>::Action {
        let (dist, _) = self.net.forward(obs, mask);
        dist.sample()
    }

    /// get the most-likely action
    pub fn mode<M: ActionMask<Value = Tensor<2>>>(&self, obs: Net::Obs, mask: M) -> <Net::Dist as Distribution>::Action {
        let (dist, _) = self.net.forward(obs, mask);
        dist.mode()
    }

    /// update the value network and policy
    pub fn update<M: ActionMask<Value = Tensor<2>>>(mut self, t: Batch<Net::Obs, <Net::Dist as Distribution>::Action, M>) -> (Self, Tensor<1>, Tensor<1>, Tensor<1>) {
        let n = t.batch_size;
        let device = t.rewards.device();
        let rewards: Vec<f32> = t.rewards.into_data().into_vec().unwrap();
        let terminated: Vec<f32> = t.terminated.into_data().into_vec().unwrap();
        let truncated: Vec<f32> = t.truncated.into_data().into_vec().unwrap();
        let (_, next_values) = self.net.forward(t.next_obss, t.next_masks);
        let next_values = next_values.detach().into_data().into_vec().unwrap();
        let mut targets = vec![0f32; n + 1];
        targets[n] = if terminated[n - 1] == 1f32 { 0f32 } else { next_values[n - 1] };

        for i in (0..n).rev() {
            targets[i] = 
            if terminated[i] != 0f32 || truncated[i] != 0f32 {
                rewards[i] + self.gamma * next_values[i] * (1f32 - terminated[i])
            } else {
                rewards[i] + self.gamma * targets[i + 1]
            };
        }
        targets.truncate(n);
        // 1. advantage
        let targets: Tensor<1> = Tensor::from_data(TensorData::new(targets, [n]), &device);
        let (dist, values) = self.net.forward(t.obss, t.masks);
        let advantages = (targets.clone() - values.clone()).detach();
        let advantages = (advantages.clone() - advantages.clone().mean()) / (advantages.var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(t.actions);
        let policy_loss = -(log_prob * advantages).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let value_loss = MseLoss.forward(values, targets, Reduction::Mean);
        // 5. backward
        let loss = policy_loss.clone() - entropy.clone() * self.c_e + value_loss.clone() * self.c_v;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.net);
        self.net = self.opt.step(self.lr, self.net, grads);

        (self, value_loss, policy_loss, entropy)
    }
}