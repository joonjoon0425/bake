//! uniform sampler
use crate::core::data::{Shaped, BatchOps};
use burn::prelude::*;
use super::{Sampler, Sample};

/// samples uniformly
/// - used for ordinary replay buffer
pub struct UniformSampler {
    device: Device,
}

impl UniformSampler {
    /// create new `UniformSampler`
    pub fn new(device: &Device) -> Self { Self { device: device.clone() } }
}

impl Sampler for UniformSampler {
    type SampleInfo = ();

    fn sample<T: BatchOps>(&self, size: usize, len: usize, storage: &Shaped<T>) -> Sample<T, ()> {
        let indices = Tensor::random([size], burn::tensor::Distribution::Uniform(0., len as f64), &self.device);

        Sample {
            sample: storage.clone().select(0, indices.clone()),
            indices,
            info: ()
        }
    }
}