//! Double Deep Q-Network
use burn::prelude::*;
use crate::deep::buffer::sampler::SampleInfo;
use crate::core::constraint::DiscreteConstraint;
use crate::core::data::{Transition, Batch, BatchOps};
use crate::deep::{
    contract::DiscreteActionValue,
    loss::{Loss, LossFn}
};

/// state for Double Deep Q-Network algorithm
#[derive(Debug, Clone)]
pub struct DoubleDQNState {
    /// discount factor
    pub gamma: f32,
    /// loss function
    pub loss_fn: LossFn,
}

/// Double Deep Q-Network loss
#[derive(Debug, Clone)]
pub struct DoubleDQNLoss {
    /// loss value for autodiff
    pub loss: Tensor<1>,
    /// td error
    pub td_error: Tensor<1>,
    /// action value mean
    pub q_mean: Tensor<1>,
}

impl Loss for DoubleDQNLoss { fn loss(&self) -> Tensor<1> { self.loss.clone() } }

/// compute the loss for double dqn algorithm. If `is_weight` is not `None` in `batch_info`, the weighted loss (eg. PER loss) is returned.
/// # Warning
/// - Here, the given `online` and `target` is moved to autodiff device
/// - The `online` will be moved to inner device when `update` is called
pub fn double_dqn_loss<Q, Constraint>(
    state: &DoubleDQNState,
    online: Q,
    target: &Q,
    batch: Batch<Transition<Q::Obs, Tensor<1, Int>, Constraint, Tensor<1>>>, batch_info: SampleInfo
) -> (Q, DoubleDQNLoss)
where
    Q: DiscreteActionValue,
    Constraint: DiscreteConstraint
{
    let online = online.train();
    let batch = batch.autodiff().destruct().0;

    let qvalues = online.action_values(batch.obs, batch.constraint);
    let qvalues = qvalues.gather(1, batch.action.unsqueeze_dim(1)).squeeze_dim::<1>(1);

    let next_qvalues_online = online.action_values(batch.next_obs.clone(), batch.next_constraint.clone());
    let argmax = next_qvalues_online.argmax(1);
    let next_qvalues_target = target.action_values(batch.next_obs, batch.next_constraint).detach();
    let next_qvalues = next_qvalues_target.gather(1, argmax).squeeze_dim::<1>(1);
    let targets = (batch.reward + state.gamma * next_qvalues * (1f32 - batch.terminated.float())).detach();

    // compute logging metrics
    let td_error = (targets.clone() - qvalues.clone()).detach();
    let q_mean = qvalues.clone().detach().mean();

    match batch_info.is_weights {
        Some(is_weights) => {
            let is_weights = is_weights.autodiff();
            let loss = (state.loss_fn.forward_no_reduction(qvalues, targets) * is_weights).mean();
            (online, DoubleDQNLoss { loss, td_error, q_mean })
        },
        None => {
            let loss = state.loss_fn.forward(qvalues, targets);
            (online, DoubleDQNLoss { loss, td_error, q_mean })
        }
    }
}