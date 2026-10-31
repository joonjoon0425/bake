use rand::prelude::*;
use burn::prelude::*;
use crate::core::data::{Batch, BatchOps, batch_ops::helper::resolve_dim};

impl<T: BatchOps> Batch<T> {
    /// create minibatch
    pub fn minibatch<'a>(&'a self, dim: isize, size: usize, rng: &mut SmallRng, device: &Device) -> impl Iterator<Item = Batch<T>> + 'a {
        let d = resolve_dim(self.bs().len(), dim).unwrap();
        let n = self.bs()[d];
        let mut perm: Vec<i64> = (0..n as i64).collect();
        perm.shuffle(rng);
        let end = n;

        MiniBatchIterator {
            counter: 0,
            end,
            size,
            dim,
            perm,
            batch: self,
            device: device.clone()
        }
    }
}

pub struct MiniBatchIterator<'a, T: BatchOps> {
    counter: usize,
    end: usize,
    size: usize,
    dim: isize,
    perm: Vec<i64>,
    batch: &'a Batch<T>,
    device: Device,
}

impl<'a, T: BatchOps> Iterator for MiniBatchIterator<'a, T> {
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter >= self.end {
            None
        } else {
            let stop = (self.counter + self.size).min(self.end);
            let idx = Tensor::<1, Int>::from_ints(&self.perm[self.counter..stop], &self.device);
            self.counter = stop;
            Some(self.batch.clone().select(self.dim, idx))
        }
    }
}