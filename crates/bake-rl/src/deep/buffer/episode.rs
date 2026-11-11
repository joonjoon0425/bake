//! Episode buffer for episodic algorithms
//! 
use burn::prelude::*;
use crate::core::data::{Shaped, BatchedTransition, BatchOps};

/// Episode buffer implementation
/// - collects trajectories
pub struct EpisodeBuffer<Obs: BatchOps, Constraint: BatchOps, Action: BatchOps, Reward: BatchOps> {
    batch: Vec<Shaped<BatchedTransition<Obs, Constraint, Action, Reward>>>,
    finished: Tensor<1, Bool>,
}

impl<Obs, Constraint, Action, Reward> EpisodeBuffer<Obs, Constraint, Action, Reward>
where
    Obs: BatchOps,
    Constraint: BatchOps,
    Action: BatchOps,
    Reward: BatchOps
{
    /// create a new `RolloutBuffer`
    pub fn new(n_envs: usize, device: &Device) -> Self {
        Self { batch: vec![], finished: Tensor::zeros([n_envs], device) }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Shaped<BatchedTransition<Obs, Constraint, Action, Reward>>) {
        self.finished.inplace(|f| f.bool_or(t.terminated.clone()).bool_or(t.truncated.clone()));
        self.batch.push(t);
    }

    /// pop all elements of buffer as batch
    pub fn pop(&mut self) -> Shaped<BatchedTransition<Obs, Constraint, Action, Reward>> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        self.finished = Tensor::zeros_like(&self.finished);
        Shaped::stack(batch)
    }

    /// checks if buffer is ready
    /// - the buffer is 'ready' when each environments has collected at least one trajectory
    /// - note that we call `into_scalar` here, which causes synchronization
    pub fn is_ready(&self) -> bool {
        self.finished.clone().all().into_scalar()
    }
}