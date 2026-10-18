//! the transition (s, a, r, s'), as a batched form
//! 
use bake_macros::Batchable;
use burn::prelude::*;
use crate::core::data::Batchable;

/// the transition (s, a, r, s'), as a batched form
#[derive(Batchable, Debug, Clone)]
pub struct Batch<Obs: Batchable, Action: Batchable, Constraint: Batchable, Reward: Batchable> {
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
    pub terminated: Tensor<1, Bool>,
    /// truncated
    pub truncated: Tensor<1, Bool>
}