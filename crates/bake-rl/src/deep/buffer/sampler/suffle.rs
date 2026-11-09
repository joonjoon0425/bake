//! shuffle sampler
use crate::core::data::{Shaped, BatchOps};
use burn::prelude::*;
use super::{EpochSampler, Sample};

/// make shuffled minibatch samples
/// # Warning
/// - currently does not drop leftovers
pub struct ShuffleSampler;

impl EpochSampler for ShuffleSampler {
    type SampleInfo = ();

    fn iter<'a, T: BatchOps>(&self, chunk_size: usize, batch: &'a Shaped<T>, device: &Device) -> impl Iterator<Item = Sample<T, ()>> + 'a {
        let len = batch.batch_shape()[0];
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
    batch: &'a Shaped<T>,
}

impl<'a, T: BatchOps> Iterator for SliceSamplerIterator<'a, T> {
    type Item = Sample<T, ()>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.end {
            None
        } else {
            let stop = (self.counter + self.size).min(self.end);
            let idx = self.perm.clone().slice(self.counter..stop);
            self.counter = stop;
            Some(Sample { sample: self.batch.clone().select(0, idx.clone()), indices: idx, info: () })
        }
    }
}