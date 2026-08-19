//! A Deep Q-Network algorithm implementation
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::Int};
use crate::{network::QNetwork, policy::EpsGreedy, types::{Batch, DiscreteMask}};
/// A Deep Q-Network algorithm Implmentation
pub struct DQNAgent<QNet>
where
    QNet: QNetwork,
{
    gamma: f32,
    opt: ModuleOptimizer,
    lr: f64,
    online: QNet,
    target: QNet,
}

impl<QNet> DQNAgent<QNet>
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
    pub fn action(&self, policy: &mut EpsGreedy, obs: QNet::Obs, barrier: Option<DiscreteMask>) -> Tensor<1, Int> {
        let qvalues = self.online.forward(obs, barrier.clone()).detach();
        policy.sample(qvalues, barrier)
    }

    /// update the online network<br>
    /// returns Self, Mean Q Values, loss, and TD-error of given batches
    pub fn update(mut self, t: Batch<QNet::Obs, Tensor<1, Int>, DiscreteMask>)
    -> (Self, Tensor<1>, Tensor<1>, Tensor<1>) {
        let qvalues = self.online.forward(t.obss, t.barriers.into());
        let qvalues = qvalues.gather(1, t.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues = self.target.forward(t.next_obss, t.next_barriers.into());
        let target = (t.rewards + self.gamma * next_qvalues.max_dim(1).squeeze_dim(1) * (1f32 - t.terminated)).detach();

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