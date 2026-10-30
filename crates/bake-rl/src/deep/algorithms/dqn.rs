//! Deep Q-Network
use burn::prelude::*;
use crate::core::buffer::sampler::SampleInfo;
use crate::core::constraint::DiscreteConstraint;
use crate::core::data::{Batch, BatchTensor, DetachBatchDim, WithBatchDim};
use crate::deep::{
    contract::DiscreteActionValue,
    loss::{Loss, LossFn}
};

/// Deep Q-Network algorithm state
#[derive(Debug, Clone)]
pub struct DQNState {
    /// discount factor
    pub gamma: f32,
    /// loss function
    pub loss_fn: LossFn,
}

/// Deep Q-Network loss
#[derive(Debug, Clone)]
pub struct DQNLoss {
    /// loss value for autodiff
    pub loss: Tensor<1>,
    /// td error
    pub td_error: Tensor<1>,
    /// action value mean
    pub q_mean: Tensor<1>,
}

impl Loss for DQNLoss { fn loss(&self) -> Tensor<1> { self.loss.clone() } }

/// compute the loss for dqn algorithm. If `is_weight` is not `None` in `batch_info`, the weighted loss (eg. PER loss) is returned.
/// # Warning
/// - Here, the given `online` and `target` is moved to autodiff device
/// - The `online` will be moved to inner device when `update` is called
pub fn dqn_loss<Q, Constraint>(state: &DQNState, online: Q, target: &Q, batch: Batch<Q::Obs, BatchTensor<1, Int>, Constraint, BatchTensor<1>>, batch_info: SampleInfo) -> (Q, DQNLoss)
where
    Q: DiscreteActionValue,
    Constraint: DiscreteConstraint
{
    let online = online.train();
    let batch = batch.autodiff();

    let qvalues = online.action_values(batch.obs, batch.constraint).detach_batch_dim().0;
    let qvalues = qvalues.gather(1, batch.action.detach_batch_dim().0.unsqueeze_dim(1)).squeeze_dim::<1>(1);

    let next_qvalues = target.action_values(batch.next_obs, batch.next_constraint);
    let targets = (batch.reward.detach_batch_dim().0 + state.gamma * next_qvalues.detach_batch_dim().0.max_dim(1).squeeze_dim(1) * (1f32 - batch.terminated.detach_batch_dim().0.float())).detach();

    let td_error = (targets.clone() - qvalues.clone()).detach();
    let q_mean = qvalues.clone().detach().mean();

    match batch_info.is_weights {
        Some(is_weights) => {
            let is_weights = is_weights.autodiff();
            let loss = (state.loss_fn.forward_no_reduction(qvalues, targets) * is_weights).mean();
            (online, DQNLoss { loss, td_error, q_mean })
        },
        None => {
            let loss = state.loss_fn.forward(qvalues, targets);
            (online, DQNLoss { loss, td_error, q_mean })
        }
    }
}