//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use crate::types::Mask;

/// A transition used for all agents
/// - contains s, a, r, s' and terminated, truncated, masks, extra
#[derive(Debug, Clone)]
pub struct Transition<M: Mask, Extra: Clone = ()> {
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

impl<M: Mask> Transition<M, ()> {
    /// adds and extra field to the basic Transition<M> type
    pub fn add_extra<Extra: Clone>(self, extra: Extra) -> Transition<M, Extra> {
        Transition::<M, Extra> {
            obs: self.obs,
            action: self.action,
            reward: self.reward,
            next_obs: self.next_obs,
            terminated: self.terminated,
            truncated: self.truncated,
            mask: self.mask,
            next_mask: self.next_mask,
            extra,
        }
    }
}