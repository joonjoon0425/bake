//! the transition (s, a, r, s'), as a batched form
//! 
use std::borrow::Cow;

use bake_rl_macros::BatchOps;
use crate::core::data::{Batch, BatchDict, BatchOps, batch::dict::{DictLeaf, Key}};
use burn::prelude::*;

/// the transition (s, a, r, s')
#[derive(BatchOps, Debug, Clone)]
pub struct Transition<Repr: TransitionRepr, Obs, Action, Constraint, Reward> {
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
    pub terminated: Repr::Flag,
    /// truncated
    pub truncated: Repr::Flag,

    /// extra values
    pub extra: Repr::Extra,
}

/// transition representation
pub trait TransitionRepr: Clone + 'static {
    /// boolean flag
    type Flag;
    /// dictionaru type for extra
    type Extra;
}

impl TransitionRepr for crate::core::Native { type Extra = (); type Flag = bool; }
impl TransitionRepr for crate::core::Burn { type Extra = BatchDict; type Flag = Tensor<1, Bool>; }

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> Batch<Transition<crate::core::Burn, Obs, Action, Constraint, Reward>> {
    /// return the current device of `Transition`
    pub fn device(&self) -> Device {
        self.leaf.terminated.device()
    }

    /// insert a new extra value with given string key
    pub fn insert_str<T: BatchOps>(&mut self, key: impl Into<Cow<'static, str>>, value: T) -> Result<Option<DictLeaf>, String> {
        self.leaf.extra.insert_str(key, value)
    }

    /// insert a new extra value with given struct key
    pub fn insert<T: BatchOps>(&mut self, key: Key<T>, value: T) -> Result<Option<T>, String> {
        self.leaf.extra.insert(key, value)
    }

    /// get a value from extra, with given string key
    pub fn get_str<T: BatchOps>(&self, key: &str) -> Result<&T, String> {
        self.leaf.extra.get_str(key)
    }

    /// get a value from extra, with given struct key
    pub fn get<T: BatchOps>(&self, key: &Key<T>) -> Result<&T, String> {
        self.leaf.extra.get(key)
    }

    /// remove a value from extra, with given string key
    pub fn remove_str<T: BatchOps>(&mut self, key: &str) -> Result<T, String> {
        self.leaf.extra.remove_str(key)
    }

    /// remove a value from extra, with given struct key
    pub fn remove<T: BatchOps>(&mut self, key: &Key<T>) -> Result<T, String> {
        self.leaf.extra.remove(key)
    }

    /// modify a value of extra, with given string key
    pub fn modify_str<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &str, f: F) -> Result<(), String> {
        self.leaf.extra.modify_str(key, f)
    }

    /// modify a value of extra, with given struct key
    pub fn modify<T: BatchOps, F: FnOnce(&mut T)>(&mut self, key: &Key<T>, f: F) -> Result<(), String> {
        self.leaf.extra.modify(key, f)
    }
}