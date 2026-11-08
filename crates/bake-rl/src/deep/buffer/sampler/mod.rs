//! sampler structs

pub mod uniform;
pub mod prioritized;
pub mod slice;

use crate::core::data::{Batch, BatchOps};
use burn::prelude::*;
use super::storage::ring::RingStorage;

/// sampler trait for replay buffers
pub trait ReplayBufferSampler {
    /// returns the sample with given batch size
    fn sample<T: BatchOps>(&self, batch_size: usize, storage: &RingStorage<T>, device: &Device) -> Batch<T>;
}

/// sampler trait for rollout buffers
pub trait RolloutBufferSampler {
    /// return the iterator of minibatch samples
    /// # Warning
    /// - samples as if all batch shapes were flattened
    fn iter<'a, T: BatchOps>(&self, chunk_size: usize, batch: &'a Batch<T>, device: &Device) -> impl Iterator<Item = Batch<T>> + 'a;
}