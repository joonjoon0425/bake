//! Replay buffer
pub mod lazy_storage;

use std::marker::PhantomData;
use crate::core::{
    buffer::sampler::{Sampler, PrioritizedSampler, PrioritizedSamplerConfig, SampleInfo, SamplerConfig, uniform::UniformSamplerConfig},
    data::{Batch, WithBatchDim},
};
use crate::core::buffer::replay::lazy_storage::LazyStorage;
use burn::prelude::*;

/// Replay buffer implementation
pub struct ReplayBuffer<S: Sampler, Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> {
    /// Lazy initialization storage
    storage: LazyStorage<Obs, Action, Constraint, Reward>,
    /// the sampler
    sampler: S,
}


impl<S, Obs, Action, Constraint, Reward> ReplayBuffer<S, Obs, Action, Constraint, Reward>
where
    S: Sampler,
    Obs: WithBatchDim,
    Action: WithBatchDim,
    Constraint: WithBatchDim,
    Reward: WithBatchDim,
{
    /// create a new ReplayBuffer (user's won't use this. Users must use the ReplayBufferConfig)
    pub fn new(capacity: usize, sampler: S) -> Self {
        Self {
            storage: LazyStorage::new(capacity),
            sampler,
        }
    }

    /// Push a givn transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Reward>) {
        let (index, n) = self.storage.push(t);
        self.sampler.after_push(index, n);
    }

    /// return the number of data in buffer
    pub fn n(&self) -> usize { return self.storage.n() }

    /// sample given amount of batches from buffer. If the buffer's length is shorter than `batch_size`, returns None.
    pub fn sample(&mut self, batch_size: usize) -> Option<(Batch<Obs, Action, Constraint, Reward>, SampleInfo)> {
        let len = self.n();
        if len < batch_size { return None; }
        Some(self.sampler.sample(batch_size, &self.storage))
    }

}

impl<Obs, Action, Constraint, Reward> ReplayBuffer<PrioritizedSampler, Obs, Action, Constraint, Reward>
where
    Obs: WithBatchDim,
    Action: WithBatchDim,
    Constraint: WithBatchDim,
    Reward: WithBatchDim,
{
    /// update the priority of elements of given indices to given priorities
    pub fn update_priority(&mut self, indices: &[usize], priorities: Tensor<1>) {
        self.sampler.update_priority(indices, priorities);
    }

    /// return the beta
    pub fn beta(&self) -> f64 { self.sampler.beta() }

    /// return the mutable reference of beta
    pub fn beta_mut(&mut self) -> &mut f64 { self.sampler.beta_mut() }
}

/// A helper struct for creating a ReplayBuffer
pub struct ReplayBufferConfig<SamplerConf: SamplerConfig, Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> {
    config: SamplerConf,
    seed: u64,
    capacity: usize,
    _p: PhantomData<(Obs, Action, Constraint, Reward)>,
}

impl<SamplerConf: SamplerConfig, Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim,> ReplayBufferConfig<SamplerConf, Obs, Action, Constraint, Reward> {
    /// create a new replay buffer
    pub fn init(self) -> ReplayBuffer<SamplerConf::SamplerType, Obs, Action, Constraint, Reward> {
        ReplayBuffer::new(self.capacity, self.config.init(self.seed, self.capacity))
    }
}

impl<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim,> ReplayBufferConfig<UniformSamplerConfig, Obs, Action, Constraint, Reward> {
    /// create a new RerplayBufferConfig with UniformSampler
    pub fn uniform(seed: u64, capacity: usize) -> Self {
        Self {
            config: UniformSamplerConfig,
            seed,
            capacity,
            _p: PhantomData
        }
    }
}

impl<Obs: WithBatchDim, Action: WithBatchDim, Constraint: WithBatchDim, Reward: WithBatchDim> ReplayBufferConfig<PrioritizedSamplerConfig, Obs, Action, Constraint, Reward> {
    /// create a new ReplayBufferConfig for PER
    pub fn prioritized(seed: u64, capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            config: PrioritizedSamplerConfig::new(alpha, beta),
            seed,
            capacity,
            _p: PhantomData
        }
    }

    /// configure the priority clip
    pub fn with_priority_clip(mut self, priority_clip: f64) -> Self {
        self.config = self.config.with_priority_clip(priority_clip);
        self
    }
}