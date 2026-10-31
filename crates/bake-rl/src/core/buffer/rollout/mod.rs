//! Rollout buffer for on-policy methods
//! 
use crate::core::data::{Batch, BatchOps, Transition};
/// Rollout buffer implementation. Currently uses AoS (Array of Structures)
pub struct RolloutBuffer<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> {
    batch: Vec<Batch<Transition<Obs, Action, Constraint, Reward>>>,
}

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> RolloutBuffer<Obs, Action, Constraint, Reward> {
    /// create a new `RolloutBuffer`
    pub fn new() -> Self {
        Self { batch: vec![] }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Batch<Transition<Obs, Action, Constraint, Reward>>) {
        self.batch.push(t);
    }

    /// pop all elements of buffer
    pub fn pop(&mut self) -> Batch<Transition<Obs, Action, Constraint, Reward>> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Batch::stack(batch)
    }
}
