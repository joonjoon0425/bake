//! rollout buffer

use crate::deep::buffer::storage::ring::RingStorage;
use crate::core::data::{Batch, BatchOps};

/// rollout buffer
pub struct RolloutBuffer<T: BatchOps> {
    time_step: usize,
    storage: RingStorage<T>,
}

impl<T: BatchOps> RolloutBuffer<T> {
    /// create new rollout buffer of given capacity, with given sampler
    pub fn new(time_step: usize) -> Self {
        Self {
            time_step,
            storage: RingStorage::new(),
        }
    }

    /// push given data into buffer
    pub fn push(&mut self, data: Batch<T>) {
        if self.storage.is_init() {
            let data = data.unsqueeze(0);
            self.storage.write(data);
        } else {
            let data = data.unsqueeze(0);
            let storage = Batch::zeros_like(0, self.time_step, &data);
            self.storage.init(storage);
            self.storage.write(data);
        }
    }

    /// return current length of the buffer
    pub fn len(&self) -> usize { self.storage.len() }

    /// checks if the buffer is full
    pub fn is_full(&self) -> bool { self.time_step == self.len() }


    /// pop the buffer
    /// # Panics
    /// - when the internal storage is not initialized
    pub fn pop(&mut self) -> Batch<T> {
        let batch = std::mem::replace(&mut self.storage, RingStorage::new());
        batch.try_into_batch().unwrap()
    }
}