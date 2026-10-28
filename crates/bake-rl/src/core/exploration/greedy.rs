//! greedy policy implementation
use crate::core::{constraint::discrete::DiscreteConstraint, data::{AttachBatchDim, BatchTensor, DetachBatchDim}};
use burn::prelude::*;

use crate::deep::contract::DiscreteActionValue;
use super::Exploration;

/// A greedy policy implementation
pub struct Greedy;

impl Exploration for Greedy {
    fn sample<Q: DiscreteActionValue>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> BatchTensor<1, Int> {
        let (qvalues, bs) = qfunc.action_values(obs, constraint).detach_batch_dim();
        qvalues.argmax(1).squeeze_dim(1).attach_batch_dim(bs)
    }
}