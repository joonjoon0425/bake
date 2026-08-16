//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use burn::Tensor;

use crate::types::{ActionMask, Batchable, NoMask};

/// Transition struct
#[derive(Debug, Clone)]
pub struct Transition<Obs: Batchable, Action: Batchable, Mask: ActionMask + Batchable = NoMask, Extra = ()> {
    pub obs: Obs,
    pub action: Action,
    pub reward: f32,
    pub next_obs: Obs,
    pub terminated: bool,
    pub truncated: bool,

    pub mask: Mask,
    pub next_mask: Mask,

    pub extra: Extra,
}
/// A Batched `Transition` struct
#[derive(Debug, Clone)]
pub struct Batch<Obs, Action, Mask = NoMask, Extra = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: Tensor<1>,
    pub next_obss: Obs,
    pub terminated: Tensor<1>,
    pub truncated: Tensor<1>,
    pub masks: Mask,
    pub next_masks: Mask,
    pub extras: Extra,

    pub batch_size: usize,
}