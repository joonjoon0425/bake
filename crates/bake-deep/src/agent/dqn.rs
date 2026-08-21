//! A Deep Q-Network algorithm implementation
use burn::{Tensor, nn::loss::{MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::Int};
use crate::{constraint::DiscreteConstraint, network::QNetwork, policy::EpsGreedy, types::Batch};
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
    pub fn action(&self, policy: &mut EpsGreedy, obs: QNet::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = self.online.forward(obs, constraint.clone()).detach();
        policy.sample(qvalues, constraint)
    }

    /// update the online network<br>
    /// returns Self, Mean Q Values, loss, and TD-error of given batches
    pub fn update(mut self, t: Batch<QNet::Obs, Tensor<1, Int>, impl DiscreteConstraint>)
    -> (Self, DQNLog) {
        let qvalues = self.online.forward(t.obss, t.constraints);
        let qvalues = qvalues.gather(1, t.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues = self.target.forward(t.next_obss, t.next_constraints).detach();
        let target = (t.rewards + self.gamma * next_qvalues.max_dim(1).squeeze_dim(1) * (1f32 - t.terminated)).detach();

        let td_error = target.clone() - qvalues.clone();
        let q_mean = qvalues.clone().mean();
        let loss = MseLoss::new().forward(qvalues, target, Reduction::Mean);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online);
        self.online = self.opt.step(self.lr, self.online, grads);

        (self, DQNLog::new(q_mean, loss, td_error))
    }

    /// sync target network and online network
    pub fn sync(&mut self) {
        self.target = self.online.clone()
    }
}

/// logging struct for DQN and DDQN
#[derive(Debug, Clone, Default)]
pub struct DQNLog {
    /// Q value mean
    pub q_mean: Option<Tensor<1>>,
    /// loss
    pub loss: Option<Tensor<1>>,
    /// td error
    pub td_error: Option<Tensor<1>>,
}

impl DQNLog {
    /// create a new log struct
    pub fn new(q_mean: Tensor<1>, loss: Tensor<1>, td_error: Tensor<1>) -> Self {
        Self {
            q_mean: q_mean.into(),
            loss: loss.into(),
            td_error: td_error.into()
        }
    }
    /// q value mean
    pub fn q_mean(&self) -> f32 { self.q_mean.clone().map_or(0f32, |q| q.into_scalar()) }
    /// loss
    pub fn loss(&self) -> f32 { self.loss.clone().map_or(0f32, |q| q.into_scalar()) }
    /// td error
    pub fn mean_td_error(&self) -> f32 { self.td_error.clone().map_or(0f32, |q| q.into_scalar()) }
}