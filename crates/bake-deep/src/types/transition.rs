//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use burn::Tensor;

use crate::types::{Batchable};

/// Transition struct
#[derive(Debug, Clone)]
pub struct Transition<Obs: Batchable, Action: Batchable, Barrier: Batchable, Extra = ()> {
    pub obs: Obs,
    pub action: Action,
    pub reward: f32,
    pub next_obs: Obs,
    pub terminated: bool,
    pub truncated: bool,

    pub barrier: Option<Barrier>,
    pub next_barrier: Option<Barrier>,

    pub extra: Extra,
}
/// A Batched `Transition` struct
#[derive(Debug, Clone)]
pub struct Batch<Obs, Action, Barrier, Extra = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: Tensor<1>,
    pub next_obss: Obs,
    pub terminated: Tensor<1>,
    pub truncated: Tensor<1>,
    pub barriers: Option<Barrier>,
    pub next_barriers: Option<Barrier>,
    pub extras: Extra,

    pub batch_size: usize,
}