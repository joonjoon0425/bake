//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use burn::{Tensor, tensor::Int};

use crate::types::{Batchable, Indexable};

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

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable> Transition<Obs, Action, Constraint> {
    pub fn add_extra<Extra: Batchable>(self, extra: Extra) -> Transition<Obs, Action, Constraint, Extra> {
        Transition {
            obs: self.obs,
            action: self.action,
            reward: self.reward,
            next_obs: self.next_obs,
            terminated: self.terminated,
            truncated: self.truncated,
            constraint: self.constraint,
            next_constraints: self.next_constraints,
            extra,
        }
    }
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

impl<Obs: Indexable, Action: Indexable, Constraint: Indexable, Extra: Indexable> Batch<Obs, Action, Constraint, Extra> {
    pub fn select(&self, idx: Tensor<1, Int>) -> Self {
        Self {
            obss: self.obss.select(idx.clone()),
            actions: self.actions.select(idx.clone()),
            rewards: self.rewards.clone().select(0, idx.clone()),
            next_obss: self.next_obss.select(idx.clone()),
            terminated: self.terminated.clone().select(0, idx.clone()),
            truncated: self.truncated.clone().select(0, idx.clone()),
            constraints: self.constraints.select(idx.clone()),
            next_constraints: self.next_constraints.select(idx.clone()),
            extras: self.extras.select(idx.clone()),
            batch_size: idx.shape()[0],
        }
    }
}