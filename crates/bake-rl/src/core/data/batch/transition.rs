//! the transition (s, a, r, s'), as a batched form
//! 
use std::borrow::Cow;

use bake_rl_macros::BatchOps;
use crate::core::data::{Batch, BatchDict, BatchOps, batch::dict::{DictLeaf, Key}};
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
    pub truncated: Tensor<1, Bool>,

    /// extra values
    pub extra: BatchDict,
}

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> Batch<Transition<Obs, Action, Constraint, Reward>> {
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