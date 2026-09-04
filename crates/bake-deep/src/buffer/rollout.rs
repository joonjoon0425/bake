//! Rollout buffer for on-policy methods
//! 

use crate::data::{Batch, Batchable};
/// Rollout buffer implementation.
pub struct RolloutBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    batch: Vec<Batch<Obs, Action, Constraint, Extra>>,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> RolloutBuffer<Obs, Action, Constraint, Extra> {
    /// create a new `RolloutBuffer`
    pub fn new() -> Self {
        Self { batch: vec![] }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        self.batch.push(t);
    }

    /// pop all elements of buffer
    pub fn pop(&mut self) -> Batch<Obs, Action, Constraint, Extra> {
        let device = &self.batch[0].rewards.device().autodiff();
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Batch::cat(batch).to_device(device)
    }
}