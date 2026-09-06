//! A batched transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use bake_macros::Batchable;
use burn::prelude::*;
use crate::data::{batchable::Batchable, extras::{ExtraContainer, Key}};
/// A Batched transition struct
#[derive(Debug, Clone, Batchable)]
pub struct Batch<Obs: Batchable, Action: Batchable, Constraint: Batchable> {
    /// The state which agent observed
    pub obss: Obs,
    /// The action which agent did
    pub actions: Action,
    /// rewards
    pub rewards: Tensor<1>,
    /// The next state after the action
    pub next_obss: Obs,
    /// true if `next_obss` is terminal state
    pub terminated: Tensor<1>,
    /// true if the environment truncated
    pub truncated: Tensor<1>,
    /// constraint for `obss`
    pub constraints: Constraint,
    /// constraint for `next_obss`
    pub next_constraints: Constraint,
    /// extra item
    pub extras: ExtraContainer,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable> Batch<Obs, Action, Constraint> {
    /// returns the device of current Batch from `rewards` member variable. Whole training loop must use one singleton of Device object.
    pub fn device(&self) -> burn::tensor::Device {
        self.rewards.device()
    }

    /// get data from extra
    pub fn get<K: Key>(&self) -> Option<&K::Value> {
        self.extras.get::<K>()
    }

    /// get the mutable reference from extra
    pub fn get_mut<K: Key>(&mut self) -> Option<&mut K::Value> {
        self.extras.get_mut::<K>()
    }

    /// insert the extra with given key
    pub fn insert<K: Key>(&mut self, value: K::Value) {
        self.extras.insert::<K>(value);
    }

    /// remove the value in extra with given key and return the removed value
    pub fn remove<K: Key>(&mut self) -> Option<K::Value> {
        self.extras.remove::<K>()
    }
}