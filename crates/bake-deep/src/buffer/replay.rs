//! Replay buffer struct for off-policy methods
//! 
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use burn::prelude::*;
use crate::data::{Batch, Batchable};

/// Replay buffer implementation
pub struct ReplayBuffer<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,
    /// for lazy initialization, we make it optional
    batch: Option<Batch<Obs, Action, Constraint, Extra>>,

    rng: SmallRng,
}

impl<Obs, Action, Constraint, Extra> ReplayBuffer<Obs, Action, Constraint, Extra>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// Create a new replay buffer. Here, the buffer memory is not allocated until first data is pushed. 
    pub fn new(seed: u64, capacity: usize) -> Self {
        Self {
            capacity,
            head: 0,
            rng: SmallRng::seed_from_u64(seed),
            batch: None
        }
    }

    /// Push a givn transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        if self.batch.is_none() {
            self.batch = Some(Batch::zeros_like(self.capacity, &t, &t.rewards.device()))
        }
        self.batch.as_mut().unwrap().assign_inplace(t, self.head);
        
        self.head = (self.head + 1) & self.capacity;
    }

    /// return the number of data in buffer
    pub fn len(&self) -> usize {
        match self.batch.len() {
            Some(b) => b,
            None => 0,
        }
    }

    /// sample given amount of batches from buffer. If the buffer's length is shorter than `batch_size` returns None. When sampling, the autodiff backend is attached.
    pub fn sample(&mut self, batch_size: usize) -> Option<Batch<Obs, Action, Constraint, Extra>> {
        let len = self.len();
        let device = &self.batch.as_ref().unwrap().rewards.device().autodiff();
        if len < batch_size { return None; }

        let indices: Vec<usize> = (0..batch_size).map(|_| self.rng.random_range(0..len)).collect();
        let indices = Tensor::from_ints(indices.as_slice(), device);

        Some(self.batch.clone().unwrap().to_device(device).select(indices))
    }

}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use crate::{buffer::replay::ReplayBuffer, constraint::Unconstrained, data::Batch};

    #[test]
    fn init_test() {
        let device = Device::default();
        let mut buffer = ReplayBuffer::new(11, 1000);
        let obs = Tensor::<2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let action = Tensor::<1, Int>::from_ints([1], &device);
        let reward = Tensor::<1>::from_floats([1.0], &device);
        let batch = Batch {
            obss: obs.clone(),
            actions: action.clone(),
            rewards: reward.clone(),
            next_obss: obs.clone(),
            constraints: Unconstrained,
            next_constraints: Unconstrained,
            terminated: reward.clone(),
            truncated: reward.clone(),
            extras: (),
        };

        buffer.push(batch.clone());
        buffer.push(batch);
    }
}