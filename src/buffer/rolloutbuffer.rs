use burn::tensor::backend::Backend;

use crate::{traits::Batchable, transition::{BatchedTransition, Transition}};

pub struct RolloutBuffer<B: Backend, Obs: Batchable, Action: Batchable, Mask: Batchable = (), Extra: Batchable = ()>{
    observations: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_observations: Vec<Obs>,
    terminated: Vec<f32>,
    truncated: Vec<f32>,

    mask: Vec<Mask>,
    next_mask: Vec<Mask>,
    extras: Vec<Extra>,

    device: B::Device,

    n: usize,
}

impl<B: Backend, Obs: Batchable, Action: Batchable, Mask: Batchable, Extra: Batchable> RolloutBuffer<B, Obs, Action, Mask, Extra> {
    pub fn new(n: usize, device: B::Device) -> Self {
        Self {
            observations: vec![],
            actions: vec![],
            rewards: vec![],
            next_observations: vec![],
            terminated: vec![],
            truncated: vec![],
            mask: vec![],
            next_mask: vec![],
            extras: vec![],

            device,
            n,
        }
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_full(&self) -> bool {
        self.len() == self.n
    }

    pub fn push(&mut self, t: Transition<B, Obs, Action, Mask, Extra>) {
        self.observations.push(t.observation);
        self.actions.push(t.action);
        self.rewards.push(t.reward);
        self.next_observations.push(t.next_observation);
        self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
        self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
        self.mask.push(t.mask);
        self.next_mask.push(t.next_mask);
        self.extras.push(t.extra);
    }

    pub fn pop(&mut self) -> BatchedTransition<B, Obs::Batched<B>, Action::Batched<B>, Mask::Batched<B>, Extra::Batched<B>> {
        let batched_steps = BatchedTransition {
            observations: Obs::batch(self.observations.clone(), &self.device),
            actions: Action::batch(self.actions.clone(), &self.device),
            rewards: f32::batch(self.rewards.clone(), &self.device),
            next_observations: Obs::batch(self.next_observations.clone(), &self.device),
            terminated: f32::batch(self.terminated.clone(), &self.device),
            truncated: f32::batch(self.truncated.clone(), &self.device),
            mask: Mask::batch(self.mask.clone(), &self.device),
            next_mask: Mask::batch(self.next_mask.clone(), &self.device),
            extras: Extra::batch(self.extras.clone(), &self.device),

            batch_size: self.len()
        };

        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_observations.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.extras.clear();

        batched_steps
    }
}