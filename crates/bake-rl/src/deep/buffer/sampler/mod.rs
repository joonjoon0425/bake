//! sampler structs

pub mod uniform;
pub mod prioritized;
pub mod suffle;

use crate::core::data::{Shaped, BatchOps};
use burn::prelude::*;

/// sample struct with additional information
pub struct Sample<T: BatchOps, Info> {
    /// batched samples
    pub sample: Shaped<T>,
    /// indices of samples
    pub indices: Tensor<1, Int>,
    /// additional information about samples
    pub info: Info,
}

/// sampler trait for replay buffers
pub trait Sampler {
    /// information about samples
    type SampleInfo;
    /// returns the sample with given batch size
    fn sample<T: BatchOps>(&self, size: usize, len: usize, storage: &Shaped<T>) -> Sample<T, Self::SampleInfo>;
    /// hook which is called after data are pushed into replay buffer
    /// # Warning
    /// - no-op basically
    fn after_push(&mut self, _indices: Tensor<1, Int>) {}
}

/// sampler trait for rollout buffers
pub trait EpochSampler {
    /// information about samples
    type SampleInfo;
    /// return the iterator of minibatch samples
    /// # Warning
    /// - samples as if all batch shapes were flattened
    fn iter<'a, T: BatchOps>(&self, chunk_size: usize, batch: &'a Shaped<T>, device: &Device) -> impl Iterator<Item = Sample<T, Self::SampleInfo>> + 'a;
}