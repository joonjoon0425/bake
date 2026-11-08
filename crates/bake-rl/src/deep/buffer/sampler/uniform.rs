//! uniform sampler
use crate::core::data::{Batch, BatchOps};
use burn::prelude::*;
use super::ReplayBufferSampler;
use super::RingStorage;
/// samples uniformly
/// - used for ordinary replay buffer
pub struct UniformSampler;

impl ReplayBufferSampler for UniformSampler {
    fn sample<T: BatchOps>(&self, size: usize, storage: &RingStorage<T>, device: &Device) -> Batch<T> {
        let len = storage.len();
        let indices = Tensor::random([size], burn::tensor::Distribution::Uniform(0., len as f64), device);
        storage.storage().unwrap().clone().select(0, indices)
    }
}