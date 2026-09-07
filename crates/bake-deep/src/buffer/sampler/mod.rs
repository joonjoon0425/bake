//! A sampler trait and implementations for buffers
//! 
use burn::prelude::*;
use crate::{buffer::replay::LazyStorage, data::{Batch, Batchable}};

/// A `Sampler` trait which all samplers for buffers must implement
pub trait Sampler {
    /// sample n elements from given storage (currently only Batch type)
    fn sample<Obs, Action, Constraint>(&mut self, sample_size: usize, storage: &LazyStorage<Obs, Action, Constraint>) -> (Batch<Obs, Action, Constraint>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable;

    /// when a new element is pushed into buffer. no-op for base
    fn on_push(&mut self, _index: usize) { }
}

/// A common trait for Sampler configuration
pub trait SamplerConfig {
    /// Sampler
    type SamplerType: Sampler;

    /// creates a sampler
    fn init(self, seed: u64, capacity: usize) -> Self::SamplerType;
}

/// A struct holding the information of samples
pub struct SampleInfo {
    /// the indices of sample as a member of given buffer
    pub indices: Vec<usize>,
    /// importance weights
    pub is_weights: Option<Tensor<1>>,
}

pub mod uniform;
pub use uniform::UniformSampler;

pub mod prioritized;
pub use prioritized::{PrioritizedSampler, PrioritizedSamplerConfig};