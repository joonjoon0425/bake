//! lazy storage type

use crate::core::data::{Batch, WithBatchDim};
use burn::prelude::*;
/// lazy-initialization storage
#[derive(Debug)]
pub struct LazyStorage<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> {
    capacity: usize,
    head: usize,
    /// for lazy initialization, we make it optional
    buffer: Option<Batch<Obs, Action, Constraint, Reward>>,
    /// the amount of data
    n: usize,
}

impl<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> LazyStorage<Obs, Action, Constraint, Reward> {
    /// create a new LazyStorage
    pub fn new(capacity: usize) -> Self { Self { capacity , head: 0, buffer: None, n: 0 } }
    /// initialize the internal buffer with given buffer
    /// # Panic
    /// - If you call init more than once
    pub fn init(&mut self, buffer: Batch<Obs, Action, Constraint, Reward>) {
        if self.buffer.is_some() { panic!("Cannot call LazyStorage::init more than once") }
        self.buffer = Some(buffer);
    }
    /// return the number of data a LazyStorage is holding
    pub fn n(&self) -> usize { self.n }
    /// Push a givn transition into buffer and return the pushed indices
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Reward>) -> (usize, usize) {
        let batch_size = *t.batch_shape().get(0).unwrap_or(&1);
        if self.buffer.is_none() {
            assert!(self.capacity % batch_size== 0, "the number of environments (or batch size) must divide the capacity of buffer.");
            if t.batch_shape().is_empty() {
                // single, empty batch data
                self.init(Batch::zeros_like(self.capacity, &t));
            } else {
                // n_envs batch data
                self.init(Batch::zeros_like(self.capacity, &t.clone().slice(0, 0..1).squeeze(0)));
            }
        }
        let index = self.head;
        self.buffer.as_mut().unwrap().assign_inplace(t, index);
        self.head = (self.head + batch_size) % self.capacity;

        if self.n < self.capacity { self.n += batch_size; }
        (index, batch_size)
    }
    /// return the data of selected indices
    pub fn select(&self, indices: Tensor<1, Int>) -> Batch<Obs, Action, Constraint, Reward> {
        self.buffer.clone().unwrap().select(0, indices)
    }
    /// return current device of buffer
    pub fn device(&self) -> Device { self.buffer.as_ref().unwrap().device() }
}