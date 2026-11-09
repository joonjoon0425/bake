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
    fn sample<T: BatchOps>(&self, size: usize, len: usize, storage: &Shaped<T>, device: &Device) -> Sample<T, Self::SampleInfo>;
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