//! Frequently used data types
//! 
use std::collections::HashMap;
use crate::constraint::Constraint;

/// A transition used for all agents
/// - contains s, a, r, s' and terminated, truncated, masks, extra
#[derive(Debug, Clone)]
pub struct Transition<C: Constraint> {
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
    pub constraint: C,
    /// action mask of next state
    pub next_constraint: C,

    /// extra field (currently only holds f32)
    pub extra: HashMap<&'static str, f32>,
}

impl<C: Constraint> Transition<C> {
    /// get a data of given key
    pub fn get(&self, key: &'static str) -> Option<f32> {
        self.extra.get(key).copied()
    }
    /// get a mutable reference of given key
    pub fn get_mut(&mut self, key: &'static str) -> Option<&mut f32> {
        self.extra.get_mut(key)
    }
    /// remove the item of give key and give the removed item
    pub fn remove(&mut self, key: &'static str) -> Option<f32> {
        self.extra.remove(key)
    }
    /// insert new key-value pair
    /// 
    /// if the given key already existed, the value is updated and previous value is returned
    /// 
    /// if the given key was not present, `None` is returned
    pub fn insert(&mut self, key: &'static str, value: f32) -> Option<f32> {
        self.extra.insert(key, value)
    }
}