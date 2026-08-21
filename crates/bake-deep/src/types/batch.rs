//! A transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use bake_macros::Batchable;
use burn::{Tensor, tensor::Int};

use crate::types::{Batchable};
/// A Batched `Transition` struct for CPU. Later when I should support GPU, I should make rewards and terminated, truncated into Tensor
#[derive(Debug, Clone, Batchable)]
pub struct Batch<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: Tensor<1>,
    pub next_obss: Obs,
    pub terminated: Tensor<1>,
    pub truncated: Tensor<1>,
    pub constraints: Constraint,
    pub next_constraints: Constraint,
    pub extras: Extra,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, OrigExtra: Batchable> Batch<Obs, Action, Constraint, OrigExtra> {
    pub fn add_extra<Extra: Batchable>(self, extras: Extra) -> Batch<Obs, Action, Constraint, Extra> {
        Batch {
            obss: self.obss,
            actions: self.actions,
            rewards: self.rewards,
            next_obss: self.next_obss,
            terminated: self.terminated,
            truncated: self.truncated,
            constraints: self.constraints,
            next_constraints: self.next_constraints,
            extras,
        }
    }
}