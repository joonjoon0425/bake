//! the transition (s, a, r, s'), as a batched form
//! 
use bake_macros::BatchOps;
use crate::core::data::{Batch, BatchOps};
use burn::prelude::*;

/// the transition (s, a, r, s'), as a batched form
#[derive(BatchOps, Debug, Clone)]
pub struct Transition<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> {
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

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> Batch<Transition<Obs, Action, Constraint, Reward>> {
    /// return the current device of `Transition`
    pub fn device(&self) -> Device {
        self.leaf.terminated.device()
    }
} 