//! A batched transition used for all agents
//! - contains s, a, r, s' and terminated, truncated, masks, extra

use bake_macros::Batchable;
use burn::{Tensor, tensor::Int};
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