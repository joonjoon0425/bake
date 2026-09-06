//! A basic sampler which treats all elements equally
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use burn::prelude::*;
use crate::{buffer::sampler::{SampleInfo, Sampler, SamplerConfig}, data::{Batch, Batchable}};

/// A basic sampler which treats all elements equally
pub struct UniformSampler {
    rng: SmallRng,
}

impl UniformSampler {
    /// create a new `UniformSampler`
    pub fn new(seed: u64) -> Self { Self { rng: SmallRng::seed_from_u64(seed) } }
}

impl Sampler for UniformSampler {
    /// # Panic
    /// panics when the length of the storage is smaller than n
    fn sample<Obs, Action, Constraint, Extra>(&mut self, n: usize, storage: &Batch<Obs, Action, Constraint, Extra>) -> (Batch<Obs, Action, Constraint, Extra>, SampleInfo)
    where
        Obs: Batchable,
        Action: Batchable,
        Constraint: Batchable,
        Extra: Batchable
    {
        let len = storage.len().unwrap();
        if len < n { panic!("Sampler received n bigger than given storage's length") }

        let indices_raw: Vec<usize> = (0..n).map(|_| self.rng.random_range(0..len)).collect();
        let indices = Tensor::from_ints(indices_raw.as_slice(), &storage.device());
        (storage.clone().select(indices), SampleInfo { indices: indices_raw , is_weights: None } )
    }
}

/// Empty config struct for UniformSampler
pub struct UniformSamplerConfig;
impl SamplerConfig for UniformSamplerConfig {
    type SamplerType = UniformSampler;
    fn init(self, seed: u64, _: usize) -> Self::SamplerType { UniformSampler::new(seed) }
}