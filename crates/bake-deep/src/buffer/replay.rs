//! Replay buffer struct for off-policy methods
//! 
use burn::tensor::Device;

use crate::{buffer::sampler::{PrioritizedSampler, PrioritizedSamplerConfig, SampleInfo, Sampler, UniformSampler}, data::{Batch, Batchable}};

/// Replay buffer implementation
pub struct ReplayBuffer<S: Sampler, Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable = ()> {
    capacity: usize,
    head: usize,
    len: usize,
    /// for lazy initialization, we make it optional
    batch: Option<Batch<Obs, Action, Constraint, Extra>>,
    /// the sampler
    sampler: S,
    /// autodiff attached device
    device: Option<Device>,
}

impl<Obs, Action, Constraint, Extra> ReplayBuffer<UniformSampler, Obs, Action, Constraint, Extra>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// create a new ReplayBuffer with uniform sampler
    pub fn uniform(seed: u64, capacity: usize) -> Self {
        Self {
            head: 0,
            len: 0,
            capacity,
            batch: None,
            device: None,
            sampler: UniformSampler::new(seed)
        }
    }
}

impl<Obs, Action, Constraint, Extra> ReplayBuffer<PrioritizedSampler, Obs, Action, Constraint, Extra>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// create a new ReplayBuffer with uniform sampler
    pub fn priority(seed: u64, capacity: usize, alpha: f64, beta: f64) -> PrioritizedSamplerConfig {
        PrioritizedSamplerConfig {
            seed,
        }
    }
}

impl<S, Obs, Action, Constraint, Extra> ReplayBuffer<S, Obs, Action, Constraint, Extra>
where
    S: Sampler,
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// Push a givn transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        if self.batch.is_none() {
            self.batch = Some(Batch::zeros_like(self.capacity, &t, &t.device()));
            self.device = Some(t.device().autodiff());
        }
        self.batch.as_mut().unwrap().assign_inplace(t, self.head);
        
        self.head = (self.head + 1) % self.capacity;

        if self.len < self.capacity { self.len += 1; }
    }

    /// return the number of data in buffer
    pub fn len(&self) -> usize { return self.len }

    /// sample given amount of batches from buffer. If the buffer's length is shorter than `batch_size`, returns None. When sampling, the autodiff backend is attached.
    pub fn sample(&mut self, batch_size: usize) -> Option<(Batch<Obs, Action, Constraint, Extra>, SampleInfo)> {
        let len = self.len();
        if len < batch_size { return None; }
        let (sample, info) = self.sampler.sample(batch_size, self.batch.as_ref().unwrap());
        Some((sample.into_autodiff(), info))
    }

}

#[cfg(test)]
mod tests {
    use burn::{prelude::*, tensor::Distribution};
    use crate::{buffer::replay::ReplayBuffer, constraint::Unconstrained, data::{Batch, Batchable}};

    #[test]
    fn init_test() {
        let device = Device::default();
        let mut buffer = ReplayBuffer::uniform(11, 1000);
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
        assert!(buffer.len() == 1);
        buffer.push(batch);
        assert!(buffer.len() == 2);
    }

    #[test]
    fn sample_test() {
        let device = Device::default();
        let mut buffer = ReplayBuffer::uniform(11, 1000);

        for _ in 0..100 {
            let obs = Tensor::<2>::random([1, 4], Distribution::Uniform(-3.0, 3.0), &device);
            let action = Tensor::<1, Int>::random([1], Distribution::Uniform(0.0, 2.0), &device);
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
            buffer.push(batch);
        }
        
        assert!(buffer.sample(1000).is_none());
        assert!(buffer.len() == 100);
        assert!(buffer.sample(64).is_some());
        assert!(buffer.sample(64).unwrap().0.len().unwrap() == 64);
    }

    #[test]
    fn wrap_around() {
        let device = Device::default();
        let mut buffer = ReplayBuffer::uniform(11, 10);

        for i in 0..11 {
            let obs = Tensor::<2>::full([1, 4], i as f32, &device);
            let action = Tensor::<1, Int>::random([1], Distribution::Uniform(0.0, 2.0), &device);
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
            buffer.push(batch);
        }

        let inner = buffer.batch.as_ref().unwrap();
        assert!(!inner.obss.clone().equal_elem(0.0).any().into_scalar::<bool>());
        println!("head: {}, {}", buffer.head, inner.obss);
        println!("{}", buffer.sample(1).unwrap().0.rewards);
    }
}