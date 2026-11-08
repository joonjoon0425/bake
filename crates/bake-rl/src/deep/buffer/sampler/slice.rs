//! slicing sampler
use crate::core::data::{Batch, BatchOps};
use burn::prelude::*;
use super::RolloutBufferSampler;
/// make sliced samples
pub struct SliceSampler;

impl RolloutBufferSampler for SliceSampler {
    fn iter<'a, T: BatchOps>(&self, chunk_size: usize, batch: &'a Batch<T>, device: &Device) -> impl Iterator<Item = Batch<T>> + 'a {
        let len = batch.batch_shape().iter().product::<usize>();
        let rand_float = Tensor::<1>::random([len], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
        let perm = rand_float.argsort(0);
        SliceSamplerIterator {
            counter: 0,
            end: len,
            size: chunk_size,
            batch,
            perm,
        }
    }
}
/// slice sample iterator
pub struct SliceSamplerIterator<'a, T: BatchOps> {
    counter: usize,
    end: usize,
    size: usize,
    perm: Tensor<1, Int>,
    batch: &'a Batch<T>,
}

impl<'a, T: BatchOps> Iterator for SliceSamplerIterator<'a, T> {
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.end {
            None
        } else {
            let stop = (self.counter + self.size).min(self.end);
            let idx = self.perm.clone().slice(self.counter..stop);
            self.counter = stop;
            Some(self.batch.clone().flatten().select(0, idx))
        }
    }
}