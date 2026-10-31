//! various samplers for buffer

use crate::core::data::{Batch, BatchOps, Transition};
use crate::core::buffer::replay::lazy_storage::LazyStorage;
use burn::prelude::*;

/// A `Sampler` trait which all samplers for buffers must implement
pub trait Sampler {
    /// sample n elements from given storage (currently only Batch type)
    fn sample<Obs, Action, Constraint, Reward>(&mut self, sample_size: usize, storage: &LazyStorage<Obs, Action, Constraint, Reward>) -> (Batch<Transition<Obs, Action, Constraint, Reward>>, SampleInfo)
    where
        Obs: BatchOps,
        Action: BatchOps,
        Constraint: BatchOps,
        Reward: BatchOps;

    /// when a new element is pushed into buffer. no-op for base
    fn after_push(&mut self, _index: usize, _n: usize) { }
}

/// A common trait for Sampler configuration
pub trait SamplerConfig {
    /// Sampler
    type SamplerType: Sampler;

    /// creates a sampler
    fn init(self, seed: u64, capacity: usize) -> Self::SamplerType;
}

/// A struct holding the information of samples
#[derive(Debug, Clone)]
pub struct SampleInfo {
    /// the indices of sample as a member of given buffer
    pub indices: Vec<usize>,
    /// importance weights
    pub is_weights: Option<Tensor<1>>,
}

pub mod uniform;
pub use uniform::{UniformSampler, UniformSamplerConfig};

pub mod prioritized;
pub use prioritized::{PrioritizedSampler, PrioritizedSamplerConfig};