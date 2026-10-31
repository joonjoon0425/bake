use rand::prelude::*;
use burn::prelude::*;
use crate::core::data::{Batch, BatchOps, batch_ops::helper::resolve_dim};

impl<T: BatchOps> Batch<T> {
    /// create minibatch
    pub fn minibatch<'a>(&'a self, dim: isize, size: usize, seed: u64, device: &Device) -> impl Iterator<Item = Batch<T>> + 'a {
        MiniBatchIterator::new(self.clone(), dim, size, seed, device)
    }
}

pub struct MiniBatchIterator<T: BatchOps> {
    counter: usize,
    end: usize,
    size: usize,
    dim: usize,
    perm: Vec<i64>,
    minibatch: Batch<T>,
    device: Device,
}

impl<T: BatchOps> MiniBatchIterator<T> {
    pub fn new(minibatch: Batch<T>, dim: isize, size: usize, seed: u64, device: &Device) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut perm: Vec<i64> = (0..minibatch.bs().iter().product::<usize>() as i64).collect();
        perm.shuffle(&mut rng);
        let dim = resolve_dim(minibatch.bs().len(), dim).expect("MiniBatchIterator::new: wrong index given");
        Self {
            counter: 0,
            end: minibatch.bs()[dim] / size,
            size,
            dim,
            perm,
            minibatch,
            device: device.clone()
        }
    }
}

impl<T: BatchOps> Iterator for MiniBatchIterator<T> {
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.counter > self.end {
            None
        } else {
            if self.counter >= self.end { return None; }
            let stop = (self.counter + self.size).min(self.end);
            let idx = Tensor::<1, Int>::from_ints(&self.perm[self.counter..stop], &self.device);
            self.counter = stop;
            Some(self.minibatch.clone().select(self.dim as isize, idx))
        }
    }
}