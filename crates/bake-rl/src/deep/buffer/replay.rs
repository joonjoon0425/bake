//! replay buffer
use burn::prelude::*;
use crate::deep::buffer::sampler::{Sampler, Sample};
use crate::deep::buffer::writer::{Writer, ring::RingWriter};
use crate::core::data::{Batch, BatchOps};

/// replay buffer
pub struct ReplayBuffer<S: Sampler, T: BatchOps> {
    capacity: usize,
    head: usize,
    len: usize,
    sampler: S,
    writer: RingWriter,
    storage: Option<Batch<T>>,
    device: Device,
}

impl<S: Sampler, T: BatchOps> ReplayBuffer<S, T> {
    /// create new replay buffer of given capacity, with given sampler
    pub fn new(sampler: S, capacity: usize, device: &Device) -> Self {
        Self {
            capacity,
            head: 0,
            len: 0,
            sampler,
            writer: RingWriter,
            device: device.clone(),
            storage: None,
        }
    }

    /// push given data into buffer
    /// # Warning
    /// - the capacity will be rounded up to ceil(capacity / n_envs) * n_envs in the first push
    /// - note that the n_envs must equal every push
    pub fn push(&mut self, data: Batch<T>) {
        let data = data.unsqueeze(0);
        let n_envs = data.batch_dim(-1);
        match &mut self.storage {
            Some(storage) => {
                let head = self.writer.index(self.capacity, self.head, 1).unwrap();
                storage.assign_inplace(0, self.head, data);
                self.head = head;
                if self.len != self.capacity { self.len += 1; }
            },
            None => {
                // round up capacity
                self.capacity = (self.capacity as f32 / n_envs as f32).ceil() as usize * n_envs;
                let mut storage = Batch::zeros_like(0, self.capacity, &data);
                let head = self.writer.index(self.capacity, self.head, 1).unwrap();
                storage.assign_inplace(0, self.head, data);
                self.head = head;
                if self.len != self.capacity { self.len += 1; }
                self.storage = Some(storage)
            }
        }
    }

    /// sample batch of given size
    /// # Returns
    /// - if `size` > current length, returns `None`
    /// - else returns sample
    pub fn sample(&self, size: usize) -> Option<Sample<T, S::SampleInfo>> {
        if size > self.len {
            None
        } else {
            Some(self.sampler.sample(size, self.len, &self.storage.as_ref().unwrap(), &self.device))
        }
    }

    /// return current length
    pub fn len(&self) -> usize { self.len }
}