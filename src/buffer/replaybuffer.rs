use std::marker::PhantomData;
use burn::{Tensor, tensor::backend::Backend};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::{traits::Batchable, transition::{BatchedTransition, Transition}};

pub struct ReplayBuffer<B: Backend, Obs: Batchable<B> + Clone, Action: Batchable<B> + Clone, Extra: Batchable<B> + Clone = ()> {
    _backend: PhantomData<B>,

    capacity: usize,
    head: usize,

    observations: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_observations: Vec<Obs>,

    terminated: Vec<f32>,
    truncated: Vec<f32>,

    extras: Vec<Extra>,

    rng: StdRng,
    device: B::Device,
}

impl<B: Backend, Obs: Batchable<B> + Clone, Action: Batchable<B> + Clone, Extra: Batchable<B> + Clone> ReplayBuffer<B, Obs, Action, Extra> {
    pub fn new(seed: u64, capacity: usize, device: B::Device) -> Self {
        let observations = Vec::with_capacity(capacity);
        let actions = Vec::with_capacity(capacity);
        let rewards = Vec::with_capacity(capacity);
        let next_observations = Vec::with_capacity(capacity);
        let terminated = Vec::with_capacity(capacity);
        let truncated = Vec::with_capacity(capacity);
        let extras = Vec::with_capacity(capacity);
        
        Self {
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
            truncated,
            extras,

            capacity,
            head: 0,
            rng: StdRng::seed_from_u64(seed),
            device,
            _backend: PhantomData
        }
    }

    pub fn push(&mut self, t: Transition<B, Obs, Action, Extra>) {
        if self.len() < self.capacity {
            self.observations.push(t.observation);
            self.actions.push(t.action);
            self.rewards.push(t.reward);
            self.next_observations.push(t.next_observation);
            self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
            self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
            self.extras.push(t.extra);
        } else {
            self.observations[self.head] = t.observation;
            self.actions[self.head] = t.action;
            self.rewards[self.head] = t.reward;
            self.next_observations[self.head] = t.next_observation;
            self.terminated[self.head] = if t.terminated { 1f32 } else { 0f32 };
            self.truncated[self.head] = if t.truncated { 1f32 } else { 0f32 };
            self.extras[self.head] = t.extra;
        }

        self.head = (self.head + 1) % self.capacity;
    }

    pub fn len(&self) -> usize { self.observations.len() }

    pub fn sample(&mut self, batch_size: usize) -> Option<BatchedTransition<B, Obs::Batched, Action::Batched, Extra::Batched>> {
        let len = self.len();

        if len < batch_size { return None; }

        let indices: Vec<usize> = (0..batch_size).map(|_| self.rng.random_range(0..len)).collect();

        let (o, a, r, no, te, tr, ex): (Vec<Obs>, Vec<Action>, Vec<f32>, Vec<Obs>, Vec<f32>, Vec<f32>, Vec<Extra>)
            = indices.iter().map(|&index| {(
                    self.observations[index].clone(),
                    self.actions[index].clone(),
                    self.rewards[index],
                    self.next_observations[index].clone(),
                    self.terminated[index],
                    self.truncated[index],
                    self.extras[index].clone())
            }).collect();

        Some(BatchedTransition {
            observations: Obs::batch(o, &self.device),
            actions: Action::batch(a, &self.device),
            rewards: Tensor::from_floats(r.as_slice(), &self.device),
            next_observations: Obs::batch(no, &self.device),
            terminated: Tensor::from_floats(te.as_slice(), &self.device),
            truncated: Tensor::from_floats(tr.as_slice(), &self.device),
            extras: Extra::batch(ex, &self.device),

            batch_size,

            _backend: PhantomData
        })
    }
}