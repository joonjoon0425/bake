//! A batched transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use bake_macros::Batchable;
use burn::prelude::*;
use crate::data::batchable::Batchable;
/// A Batched transition struct
#[derive(Debug, Clone, Batchable)]
pub struct Batch<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
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
    pub extras: Extra,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> Batch<Obs, Action, Constraint, Extra> {
    /// returns the device of current Batch from `rewards` member variable. Whole training loop must use one singleton of Device object.
    pub fn device(&self) -> burn::tensor::Device {
        self.rewards.device()
    }

    /// user can modifiy the extra solt using the given mapping function
    pub fn map_extra<ModifiedExtra: Batchable, F: Fn(Extra) -> ModifiedExtra>(self, f: F) -> Batch<Obs, Action, Constraint, ModifiedExtra> {
        let modified = f(self.extras);
        Batch {
            obss: self.obss,
            actions: self.actions,
            rewards: self.rewards,
            next_obss: self.next_obss,
            terminated: self.terminated,
            truncated: self.truncated,
            constraints: self.constraints,
            next_constraints: self.next_constraints,
            extras: modified,
        }
    }
}