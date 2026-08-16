//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

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