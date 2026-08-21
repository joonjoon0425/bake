use crate::types::{Batch, Batchable};

/// Rollout Buffer Implementation. Also works as Episode Buffer if n = None is given
pub struct RolloutBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()>{
    batch: Vec<Batch<Obs, Action, Constraint, Extra>>,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> RolloutBuffer<Obs, Action, Constraint, Extra> {
    pub fn new() -> Self {
        Self { batch: vec![] }
    }

    pub fn len(&self) -> usize {
        self.batch.len()
    }

    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        self.batch.push(t);
    }

    pub fn pop(&mut self) -> Batch<Obs, Action, Constraint, Extra> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Batch::concat(batch)
    }
}