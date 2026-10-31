//! lazy storage type

use crate::core::data::{Transition, BatchOps, Batch};
use burn::prelude::*;
/// lazy-initialization storage
#[derive(Debug)]
pub struct LazyStorage<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> {
    capacity: usize,
    head: usize,
    /// for lazy initialization, we make it optional
    buffer: Option<Batch<Transition<Obs, Action, Constraint, Reward>>>,
    /// the amount of data
    n: usize,
}

impl<Obs: BatchOps, Action: BatchOps, Constraint: BatchOps, Reward: BatchOps> LazyStorage<Obs, Action, Constraint, Reward> {
    /// create a new LazyStorage
    pub fn new(capacity: usize) -> Self { Self { capacity , head: 0, buffer: None, n: 0 } }
    /// initialize the internal buffer with given buffer
    /// # Panic
    /// - If you call init more than once
    pub fn init(&mut self, buffer: Batch<Transition<Obs, Action, Constraint, Reward>>) {
        if self.buffer.is_some() { panic!("Cannot call LazyStorage::init more than once") }
        self.buffer = Some(buffer);
    }
    /// return the number of data a LazyStorage is holding
    pub fn n(&self) -> usize { self.n }
    /// Push a givn transition into buffer and return the pushed indices
    pub fn push(&mut self, t: Batch<Transition<Obs, Action, Constraint, Reward>>) -> (usize, usize) {
        let batch_size = *t.bs().get(0).unwrap_or(&1);
        if self.buffer.is_none() {
            assert!(self.capacity % batch_size== 0, "the number of environments (or batch size) must divide the capacity of buffer.");
            self.init(Batch::zeros_like(0, self.capacity, &t));
        }
        let index = self.head;
        self.buffer.as_mut().unwrap().assign_inplace(0, index, t);
        self.head = (self.head + batch_size) % self.capacity;

        if self.n < self.capacity { self.n += batch_size; }
        (index, batch_size)
    }
    /// return the data of selected indices
    pub fn select(&self, indices: Tensor<1, Int>) -> Batch<Transition<Obs, Action, Constraint, Reward>> {
        self.buffer.clone().unwrap().select(0, indices)
    }
    /// return current device of buffer
    pub fn device(&self) -> Device { self.buffer.as_ref().unwrap().device() }
}