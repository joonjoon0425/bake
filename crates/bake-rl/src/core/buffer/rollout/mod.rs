//! Rollout buffer for on-policy methods
//! 
use crate::core::data::{Batch, WithBatchDim};
/// Rollout buffer implementation. Currently uses AoS (Array of Structures)
pub struct RolloutBuffer<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> {
    batch: Vec<Batch<Obs, Action, Constraint, Reward>>,
}

impl<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> RolloutBuffer<Obs, Action, Constraint, Reward> {
    /// create a new `RolloutBuffer`
    pub fn new() -> Self {
        Self { batch: vec![] }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Reward>) {
        self.batch.push(t);
    }

    /// pop all elements of buffer
    pub fn pop(&mut self) -> Batch<Obs, Action, Constraint, Reward> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Batch::stack(batch)
    }
}
