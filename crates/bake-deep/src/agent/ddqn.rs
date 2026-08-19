//! A Deep Q-Network algorithm implementation
use burn::{Tensor, nn::loss::{HuberLoss, HuberLossConfig, MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::Int};
use crate::{constraint::DiscreteConstraint, network::QNetwork, policy::EpsGreedy, types::Batch};
/// A Deep Q-Network algorithm Implmentation
pub struct DDQNAgent<QNet>
where
    QNet: QNetwork,
{
    gamma: f32,
    opt: ModuleOptimizer,
    lr: f64,
    online: QNet,
    target: QNet,
}

impl<QNet> DDQNAgent<QNet>
where 
    QNet: QNetwork,
{
    /// Create a new `DQNAgent`
    pub fn new(gamma: f32, qnet: QNet, lr: f64, opt: ModuleOptimizer) -> Self {
        let target = qnet.clone();
        Self {
            gamma,
            online: qnet,
            target,
            lr,
            opt,
        }
    }

    /// Sample an action using given policy
    pub fn action(&self, policy: &mut EpsGreedy, obs: QNet::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = self.online.forward(obs, constraint.clone()).detach();
        policy.sample(qvalues, constraint)
    }

    /// update the online network<br>
    /// returns Self, Mean Q Values, loss, and TD-error of given batches
    pub fn update(mut self, t: Batch<QNet::Obs, Tensor<1, Int>, impl DiscreteConstraint>)
    -> (Self, Tensor<1>, Tensor<1>, Tensor<1>) {
        let qvalues = self.online.forward(t.obss, t.constraints);
        let qvalues = qvalues.gather(1, t.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues_online = self.online.forward(t.next_obss.clone(), t.next_constraints.clone()).detach();
        let argmax = next_qvalues_online.argmax(1);
        let next_qvalues_target = self.target.forward(t.next_obss, t.next_constraints).detach();
        let next_qvalues = next_qvalues_target.gather(1, argmax).squeeze_dim::<1>(1);
        let target = (t.rewards + self.gamma * next_qvalues * (1f32 - t.terminated)).detach();

        let td_error = target.clone() - qvalues.clone();
        let q_mean = qvalues.clone().mean();
        let loss = MseLoss::new().forward(qvalues, target, Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);
        self.online = self.opt.step(self.lr, self.online, grads);

        (self, q_mean, loss, td_error)
    }

    /// sync target network and online network
    pub fn sync(&mut self) {
        self.target = self.online.clone()
    }
}