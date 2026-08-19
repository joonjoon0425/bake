//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use burn::Tensor;

use crate::types::{Batchable};

/// Transition struct
#[derive(Debug, Clone)]
pub struct Transition<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra = ()> {
    pub obs: Obs,
    pub action: Action,
    pub reward: f32,
    pub next_obs: Obs,
    pub terminated: bool,
    pub truncated: bool,

    pub constraint: Constraint,
    pub next_constraints: Constraint,

    pub extra: Extra,
}
/// A Batched `Transition` struct
#[derive(Debug, Clone)]
pub struct Batch<Obs, Action, Constraint, Extra = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: Tensor<1>,
    pub next_obss: Obs,
    pub terminated: Tensor<1>,
    pub truncated: Tensor<1>,
    pub constraints: Constraint,
    pub next_constraints: Constraint,
    pub extras: Extra,

    pub batch_size: usize,
}