use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::types::{Batch, Batchable};

/// Replay Buffer Implementation
pub struct ReplayBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,

    batch: Vec<Batch<Obs, Action, Constraint, Extra>>,

    rng: StdRng,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> ReplayBuffer<Obs, Action, Constraint, Extra> {
    pub fn new(seed: u64, capacity: usize) -> Self {
        let batch = Vec::with_capacity(capacity);
        
        Self {
            batch,
            capacity,
            head: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        if self.len() < self.capacity {
            self.batch.push(t);
        } else {
            self.batch[self.head] = t;
        }

        self.head = (self.head + 1) % self.capacity;
    }

    pub fn len(&self) -> usize { self.batch.len() }

    pub fn sample(&mut self, batch_size: usize) -> Option<Batch<Obs, Action, Constraint, Extra>> {
        let len = self.len();

        if len < batch_size { return None; }

        let indices: Vec<usize> = (0..batch_size).map(|_| self.rng.random_range(0..len)).collect();

        let selected: Vec<Batch<Obs, Action, Constraint, Extra>> = indices.iter().map(|&i| self.batch[i].clone()).collect();
        Some(Batch::concat(selected))
    }
}