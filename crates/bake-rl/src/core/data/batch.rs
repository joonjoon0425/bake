//! the transition (s, a, r, s'), as a batched form
//! 
use bake_macros::WithBatchDim;
use burn::prelude::*;

use crate::core::data::batch_dim::*;
/// the transition (s, a, r, s'), as a batched form
#[derive(WithBatchDim, Debug, Clone)]
pub struct Batch<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> {
    /// observation
    pub obs: Obs,
    /// action
    pub action: Action,
    /// reward
    pub reward: Reward,
    /// next observation
    pub next_obs: Obs,

    /// constraint
    pub constraint: Constraint,
    /// next constraint
    pub next_constraint: Constraint,

    /// terminated
    pub terminated: BatchTensor<1, Bool>,
    /// truncated
    pub truncated: BatchTensor<1, Bool>
}