//! replay buffer
use burn::prelude::*;
use crate::deep::buffer::{sampler::ReplayBufferSampler, storage::ring::RingStorage};
use crate::core::data::{Batch, BatchOps};

/// replay buffer
pub struct ReplayBuffer<Sampler: ReplayBufferSampler, T: BatchOps> {
    transition_step: usize,
    sampler: Sampler,
    storage: RingStorage<T>,
    device: Device,
}

impl<Sampler: ReplayBufferSampler, T: BatchOps> ReplayBuffer<Sampler, T> {
    /// create new replay buffer of given capacity, with given sampler
    pub fn new(sampler: Sampler, transition_step: usize, device: &Device) -> Self {
        Self {
            sampler,
            transition_step,
            device: device.clone(),
            storage: RingStorage::new(),
        }
    }

    /// push given data into buffer
    pub fn push(&mut self, mut data: Batch<T>) {
        if self.storage.is_init() {
            if data.batch_shape().is_empty() {
                data = data.unsqueeze(0);
            }
            self.storage.write(data);
            
        } else {
            if data.batch_shape().is_empty() {
                data = data.unsqueeze(0);
            }
            let storage = Batch::zeros_like(0, self.transition_step, &data);
            self.storage.init(storage);
            self.storage.write(data);            
        }
    }

    /// sample batch of given size
    /// - if `size` > current length, returns `None`
    pub fn sample(&self, size: usize) -> Option<Batch<T>> {
        if size > self.storage.len() {
            None
        } else {
            Some(self.sampler.sample(size, &self.storage, &self.device))
        }
    }
}