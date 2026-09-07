//! Replay buffer struct for off-policy methods
//! 
use std::marker::PhantomData;
use burn::prelude::*;
use crate::{buffer::sampler::{PrioritizedSampler, PrioritizedSamplerConfig, SampleInfo, Sampler, SamplerConfig, uniform::UniformSamplerConfig}, data::{Batch, Batchable}};

/// Replay buffer implementation
pub struct ReplayBuffer<S: Sampler, Obs: Batchable, Action: Batchable, Constraint: Batchable> {
    /// Lazy initialization storage
    storage: LazyStorage<Obs, Action, Constraint>,
    /// the sampler
    sampler: S,
}

/// Storage which holds the information of buffer
#[derive(Debug)]
pub struct LazyStorage<Obs: Batchable, Action: Batchable, Constraint: Batchable> {
    capacity: usize,
    head: usize,
    /// for lazy initialization, we make it optional
    buffer: Option<Batch<Obs, Action, Constraint>>,
    /// the amount of data
    n: usize,
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable> LazyStorage<Obs, Action, Constraint> {
    /// create a new LazyStorage
    pub fn new(capacity: usize) -> Self { Self { capacity , head: 0, buffer: None, n: 0 } }
    /// initialize the internal buffer with given buffer
    /// # Panic
    /// - If you call init more than once
    pub fn init(&mut self, buffer: Batch<Obs, Action, Constraint>) {
        if self.buffer.is_some() { panic!("Cannot call LazyStorage::init more than once") }
        self.buffer = Some(buffer);
    }
    /// return the number of data a LazyStorage is holding
    pub fn n(&self) -> usize { self.n }
    /// Push a givn transition into buffer and return the pushed index
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint>) -> usize {
        if self.buffer.is_none() {
            self.init(Batch::zeros_like(self.capacity, &t, &t.device()));
        }
        let index = self.head;
        self.buffer.as_mut().unwrap().assign_inplace(t, index);
        self.head = (self.head + 1) % self.capacity;

        if self.n < self.capacity { self.n += 1; }
        index
    }
    /// return the data of selected indices
    pub fn select(&self, indices: Tensor<1, Int>) -> Batch<Obs, Action, Constraint> {
        self.buffer.clone().unwrap().select(indices)
    }
    /// return current device of buffer
    pub fn device(&self) -> Device { self.buffer.as_ref().unwrap().device() }
}

impl<S, Obs, Action, Constraint> ReplayBuffer<S, Obs, Action, Constraint>
where
    S: Sampler,
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
{
    /// create a new ReplayBuffer (user's won't use this. Users must use the ReplayBufferConfig)
    pub fn new(capacity: usize, sampler: S) -> Self {
        Self {
            storage: LazyStorage::new(capacity),
            sampler,
        }
    }

    /// Push a givn transition into buffer
    pub fn push(&mut self, t: Batch<Obs, Action, Constraint>) {
        self.sampler.after_push(self.storage.push(t));
    }

    /// return the number of data in buffer
    pub fn n(&self) -> usize { return self.storage.n }

    /// sample given amount of batches from buffer. If the buffer's length is shorter than `batch_size`, returns None.
    pub fn sample(&mut self, batch_size: usize) -> Option<(Batch<Obs, Action, Constraint>, SampleInfo)> {
        let len = self.n();
        if len < batch_size { return None; }
        Some(self.sampler.sample(batch_size, &self.storage))
    }

}

impl<Obs, Action, Constraint> ReplayBuffer<PrioritizedSampler, Obs, Action, Constraint>
where
    Obs: Batchable,
    Action: Batchable,
    Constraint: Batchable,
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
pub struct ReplayBufferConfig<SamplerConf: SamplerConfig, Obs: Batchable, Action: Batchable, Constraint: Batchable> {
    config: SamplerConf,
    seed: u64,
    capacity: usize,
    _p: PhantomData<(Obs, Action, Constraint)>,
}

impl<SamplerConf: SamplerConfig, Obs: Batchable, Action: Batchable, Constraint: Batchable> ReplayBufferConfig<SamplerConf, Obs, Action, Constraint> {
    /// create a new replay buffer
    pub fn init(self) -> ReplayBuffer<SamplerConf::SamplerType, Obs, Action, Constraint> {
        ReplayBuffer::new(self.capacity, self.config.init(self.seed, self.capacity))
    }
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable> ReplayBufferConfig<UniformSamplerConfig, Obs, Action, Constraint> {
    /// create a new RerplayBufferConfig with UniformSampler
    pub fn uniform(seed: u64, capacity: usize) -> Self {
        Self {
            config: UniformSamplerConfig,
            seed,
            capacity,
            _p: PhantomData
        }
    }
}

impl<Obs: Batchable, Action: Batchable, Constraint: Batchable> ReplayBufferConfig<PrioritizedSamplerConfig, Obs, Action, Constraint> {
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
}

#[cfg(test)]
mod tests {
    use burn::{prelude::*, tensor::Distribution};
    use crate::{buffer::replay::ReplayBufferConfig, constraint::Unconstrained, data::{Batch, Batchable, extras::ExtraContainer}};

    #[test]
    fn init_test() {
        let device = Device::default();
        let mut buffer = ReplayBufferConfig::uniform(11, 1000).init();
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
            extras: ExtraContainer::new(),
        };

        buffer.push(batch.clone());
        assert!(buffer.n() == 1);
        buffer.push(batch);
        assert!(buffer.n() == 2);
    }

    #[test]
    fn sample_test() {
        let device = Device::default();
        let mut buffer = ReplayBufferConfig::uniform(11, 1000).init();

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
                extras: ExtraContainer::new(),
            };
            buffer.push(batch);
        }
        
        assert!(buffer.sample(1000).is_none());
        assert!(buffer.n() == 100);
        assert!(buffer.sample(64).is_some());
        assert!(buffer.sample(64).unwrap().0.len().unwrap() == 64);
    }

    #[test]
    fn wrap_around() {
        let device = Device::default();
        let mut buffer = ReplayBufferConfig::uniform(11, 10).init();

        for i in 0..15 {
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
                extras: ExtraContainer::new(),
            };
            buffer.push(batch);
        }

        let inner = buffer.storage.buffer.as_ref().unwrap();
        assert!(!inner.obss.clone().equal_elem(0.0).any().into_scalar::<bool>());
        println!("head: {}, {}", buffer.storage.head, inner.obss);
        println!("{}", buffer.sample(1).unwrap().0.rewards);
    }

    #[test]
    fn one_push_same_sample_test() {
        let device = Device::default();
        let mut buffer = ReplayBufferConfig::uniform(11, 10).init();
        let obs = Tensor::<2>::full([1, 4], 5 as f32, &device);
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
            extras: ExtraContainer::new(),
        };
        buffer.push(batch);

        assert!(buffer.sample(2).is_none());
        let (sample1, _) = buffer.sample(1).unwrap();
        let (sample2, _) = buffer.sample(1).unwrap();
        let (sample3, _) = buffer.sample(1).unwrap();

        assert!(sample1.obss.clone().equal(sample2.obss.clone()).all().into_scalar::<bool>());
        assert!(sample2.obss.clone().equal(sample3.obss.clone()).all().into_scalar::<bool>());
        assert!(sample3.obss.clone().equal(sample1.obss.clone()).all().into_scalar::<bool>());
    }
}