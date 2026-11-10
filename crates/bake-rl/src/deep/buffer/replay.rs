//! replay buffer
use burn::prelude::*;
use crate::deep::buffer::sampler::{Sampler, Sample};
use crate::deep::buffer::sampler::uniform::UniformSampler;
use crate::deep::buffer::sampler::prioritized::PrioritizedSampler;
use crate::deep::buffer::writer::{Writer, ring::RingWriter};
use crate::core::data::{Shaped, BatchOps};

/// replay buffer
pub struct ReplayBuffer<S: Sampler, T: BatchOps> {
    capacity: usize,
    head: usize,
    len: usize,
    sampler: S,
    device: Device,
    writer: RingWriter,
    storage: Option<Shaped<T>>,
}

impl<S: Sampler, T: BatchOps> ReplayBuffer<S, T> {
    fn new(sampler: S, capacity: usize, device: &Device) -> Self {
        Self {
            capacity,
            head: 0,
            len: 0,
            sampler,
            device: device.clone(),
            writer: RingWriter,
            storage: None,
        }
    }

    /// push given data into buffer
    pub fn push(&mut self, data: Shaped<T>) {
        let n_envs = data.batch_dim(-1);
        let head = self.writer.index(self.capacity, self.head, n_envs).unwrap();
        let indices = Tensor::arange(self.head as i64..head as i64, &self.device);
        match &mut self.storage {
            Some(storage) => {
                storage.assign_inplace(0, self.head, data);
                self.head = head;
                if self.len != self.capacity { self.len += n_envs; }
            },
            None => {
                let mut storage = Shaped::zeros_like(0, self.capacity, &data);
                storage.assign_inplace(0, self.head, data);
                self.head = head;
                if self.len != self.capacity { self.len += n_envs; }
                
                self.storage = Some(storage)
            }
        }
        
        self.sampler.after_push(indices);
    }

    /// sample batch of given size
    /// # Returns
    /// - if `size` > current length, returns `None`
    /// - else returns sample
    pub fn sample(&self, size: usize) -> Option<Sample<T, S::SampleInfo>> {
        if size > self.len {
            None
        } else {
            Some(self.sampler.sample(size, self.len, &self.storage.as_ref().unwrap()))
        }
    }

    /// return current length
    pub fn len(&self) -> usize { self.len }
}

impl<T: BatchOps> ReplayBuffer<UniformSampler, T> {
    /// create replay buffer with uniform sampler
    /// # Warning
    /// - the capacity will be rounded up to ceil(capacity / n_envs) * n_envs
    /// - note that the n_envs must equal every push
    pub fn with_uniform_sampler(capacity: usize, n_envs: usize, device: &Device) -> Self {
        let capacity = capacity.div_ceil(n_envs) * n_envs;
        Self::new(UniformSampler::new(device), capacity, device)
    }
}

impl<T: BatchOps> ReplayBuffer<PrioritizedSampler, T> {
    /// create replay buffer with prioritized sampler
    /// # Warning
    /// - the capacity will be rounded up to ceil(capacity / n_envs) * n_envs
    /// - note that the n_envs must equal every push
    pub fn with_prioritized_sampler(alpha: f32, beta: f32, priority_clip: Option<f32>, capacity: usize, n_envs: usize, device: &Device) -> Self {
        let capacity = capacity.div_ceil(n_envs) * n_envs;
        Self::new(PrioritizedSampler::new(alpha, beta, capacity, priority_clip, device), capacity, device)
    }

    /// update priorites
    pub fn update_priorites(&mut self, indices: Tensor<1, Int>, priorities: Tensor<1>) {
        self.sampler.update(indices, priorities);
    }
}