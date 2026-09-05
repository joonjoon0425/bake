//! Replay buffer struct for off-policy methods
//! 
use std::marker::PhantomData;
use burn::prelude::*;
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

impl<S, Obs, Action, Constraint, Extra> ReplayBuffer<S, Obs, Action, Constraint, Extra>
where
    S: Sampler,
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// create a new ReplayBuffer (user's won't use this. Users must use the ReplayBufferConfig)
    pub fn new(capacity: usize, sampler: S) -> Self {
        Self {
            batch: None,
            device: None,
            head: 0,
            len: 0,
            capacity,
            sampler,
        }
    }

    /// Push a givn transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint, Extra>) {
        if self.batch.is_none() {
            self.batch = Some(Batch::zeros_like(self.capacity, &t, &t.device()));
            self.device = Some(t.device().autodiff());
        }
        self.batch.as_mut().unwrap().assign_inplace(t, self.head);
        self.sampler.on_push(self.head);
        self.head = (self.head + 1) % self.capacity;

        if self.len < self.capacity { self.len += 1; }
    }

    /// return the number of data in buffer
    pub fn len(&self) -> usize { return self.len }

    /// sample given amount of batches from buffer. If the buffer's length is shorter than `batch_size`, returns None. When sampling, the autodiff backend is attached.
    pub fn sample(&mut self, batch_size: usize) -> Option<(Batch<Obs, Action, Constraint, Extra>, SampleInfo)> {
        let len = self.len();
        if len < batch_size { return None; }
        let (sample, mut info) = self.sampler.sample(batch_size, self.batch.as_ref().unwrap());
        info.is_weights = info.is_weights.map(|w| w.into_autodiff());
        Some((sample.into_autodiff(), info))
    }

}

impl<Obs, Action, Constraint, Extra> ReplayBuffer<PrioritizedSampler, Obs, Action, Constraint, Extra>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
    Extra: Batchable
{
    /// update the priority of elements of given indices to given priorities
    pub fn update_priority(&mut self, indices: &[usize], priorities: Tensor<1>) {
        self.sampler.update_priority(indices, priorities);
    }

    /// return the beta
    pub fn beta(&self) -> f64 { self.sampler.beta() }

    /// return the mutable reference of beta
    pub fn beta_mut(&mut self) -> &mut f64 { self.sampler.beta_mut() }
}

/// A helper struct for creating a ReplayBuffer
pub struct ReplayBufferConfig<SamplerConfig, Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> {
    config: SamplerConfig,
    seed: u64,
    capacity: usize,
    _p: PhantomData<(Obs, Action, Constraint, Extra)>,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> ReplayBufferConfig<(), Obs, Action, Constraint, Extra> {
    /// create a new RerplayBuffer with UniformSampler
    pub fn uniform(seed: u64, capacity: usize) -> ReplayBuffer<UniformSampler, Obs, Action, Constraint, Extra> {
        ReplayBuffer::new(capacity, UniformSampler::new(seed))
    }
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable, Extra: Batchable> ReplayBufferConfig<PrioritizedSamplerConfig, Obs, Action, Constraint, Extra> {
    /// create a new ReplayBufferConfig for PER
    pub fn prioritized(seed: u64, capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            config: PrioritizedSamplerConfig::new(alpha, beta),
            seed,
            capacity,
            _p: PhantomData
        }
    }

    /// configure the priority clip
    pub fn with_priority_clip(mut self, priority_clip: f64) -> Self {
        self.config = self.config.with_priority_clip(priority_clip);
        self
    }

    /// compute the maximum priority from current buffer if `true` is given for `flag`. Else, the maximum priority stays same as the highest priority ever sampled.
    pub fn with_max_priority_within_buffer(mut self, flag: bool) -> Self {
        self.config = self.config.with_max_priority_within_buffer(flag);
        self
    }

    /// create a new ReplayBuffer with PrioritizedSampler
    pub fn init(self) -> ReplayBuffer<PrioritizedSampler, Obs, Action, Constraint, Extra> {
        ReplayBuffer::new(self.capacity, self.config.init(self.seed, self.capacity))
    }
}

#[cfg(test)]
mod tests {
    use burn::{prelude::*, tensor::Distribution};
    use crate::{buffer::replay::ReplayBufferConfig, constraint::Unconstrained, data::{Batch, Batchable}};

    #[test]
    fn init_test() {
        let device = Device::default();
        let mut buffer = ReplayBufferConfig::uniform(11, 1000);
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
        let mut buffer = ReplayBufferConfig::uniform(11, 1000);

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
        let mut buffer = ReplayBufferConfig::uniform(11, 10);

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