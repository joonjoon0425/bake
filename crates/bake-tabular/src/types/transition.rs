//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use crate::types::Mask;

/// A transition used for all agents
/// - contains s, a, r, s' and terminated, truncated, masks, extra
pub struct Transition<M: Mask, Extra = ()> {
    /// current state (s)
    pub obs: usize,
    /// the action which agent chose (a)
    pub action: usize,
    /// reward from env
    pub reward: f32,
    /// next state (s')
    pub next_obs: usize,
    /// next state (s') is in terminal states
    pub terminated: bool,
    /// the env truncated (usually timeout) (different from terminated, since truncate should still be used at bootstrapping, etc.)
    pub truncated: bool,

    /// action mask of current state
    pub mask: M,
    /// action mask of next state
    pub next_mask: M,

    /// extra field
    pub extra: Extra
}