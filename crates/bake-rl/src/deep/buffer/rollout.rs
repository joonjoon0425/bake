//! rollout buffer

use crate::deep::buffer::writer::{Writer, linear::LinearWriter};
use crate::core::data::{Batch, BatchOps};

/// rollout buffer
pub struct RolloutBuffer<T: BatchOps> {
    rollout_size: usize,
    head: usize,
    storage: Option<Batch<T>>,
    writer: LinearWriter,
}

impl<T: BatchOps> RolloutBuffer<T> {
    /// create new rollout buffer of given capacity
    pub fn new(rollout_size: usize) -> Self {
        Self {
            rollout_size,
            head: 0,
            storage: None,
            writer: LinearWriter,
        }
    }

    /// push given data into buffer
    pub fn push(&mut self, data: Batch<T>) {
        let data = data.unsqueeze(0);
        let head = self.writer.index(self.rollout_size, self.head, 1).unwrap();
        match &mut self.storage {
            Some(storage) => {
                storage.assign_inplace(0, self.head, data);
            },
            None => {
                let mut storage = Batch::zeros_like(0, self.rollout_size, &data);
                storage.assign_inplace(0, self.head, data);
                self.storage = Some(storage);
            }
        }
        self.head = head;
    }

    /// return current length of the buffer
    pub fn len(&self) -> usize { self.head }

    /// checks if the buffer is full
    pub fn is_full(&self) -> bool { self.rollout_size == self.len() }

    /// pop the buffer
    /// # Panics
    /// - when the internal storage empty
    pub fn pop(&mut self) -> Batch<T> {
        self.head = 0;
        self.storage.take().unwrap()
    }
}