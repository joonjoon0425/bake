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
    is_full: bool,
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
            is_full: false,
            sampler,
            writer: RingWriter,
            device: device.clone(),
            storage: None,
        }
    }

    /// push given data into buffer
    pub fn push(&mut self, mut data: Batch<T>) {
        if data.batch_shape().is_empty() { data = data.unsqueeze(0); }
        let data_len = data.batch_shape()[0];
        match &mut self.storage {
            Some(storage) => {
                let (head, _) = self.writer.index(self.capacity, self.head, data_len).unwrap();
                if !self.is_full && self.head >= head { self.is_full = true; self.len = self.capacity; }
                let data_head = data.clone().slice(0, 0..self.capacity - self.head);
                let data_tail = data.slice(0, self.capacity - self.head..data_len);
                storage.assign_inplace(0, self.head, data_head);
                storage.assign_inplace(0, 0, data_tail);
                self.head = head;
            },
            None => {        
                let mut storage = Batch::zeros_like(0, self.capacity, &data);
                let (head, indices) = self.writer.index(self.capacity, self.head, data_len).unwrap();
                self.head = head;
                self.len = head;
                storage.assign_inplace(0, indices[0], data);
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