use std::marker::PhantomData;
use burn::tensor::backend::Backend;
use crate::{traits::Batchable, transition::{BatchedTransition, Transition}};

pub struct EpisodeBuffer<B: Backend, Obs: Batchable, Action: Batchable, Extra: Batchable = ()> {
    observations: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_observations: Vec<Obs>,
    terminated: Vec<f32>,
    truncated: Vec<f32>,
    extras: Vec<Extra>,

    device: B::Device,
}

impl<B: Backend, Obs: Batchable, Action: Batchable, Extra: Batchable> EpisodeBuffer<B, Obs, Action, Extra> {
    pub fn new(device: B::Device) -> Self {
        Self {
            observations: vec![],
            actions: vec![],
            rewards: vec![],
            next_observations: vec![],
            terminated: vec![],
            truncated: vec![],
            extras: vec![],

            device,
        }
    }

    pub fn len(&self) -> usize { self.observations.len() }

    pub fn push(&mut self, t: Transition<B, Obs, Action, Extra>) {
        self.observations.push(t.observation);
        self.actions.push(t.action);
        self.rewards.push(t.reward);
        self.next_observations.push(t.next_observation);
        self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
        self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
        self.extras.push(t.extra);
    }

    // gotta check here: Which is more cheaper? To clone the data and not allocating again?
    // or move the whole data and make a new empty episode buffer?
    // copy vs dynamic allocation

    // 1. copy & clear 

    pub fn pop(&mut self) -> BatchedTransition<B, Obs::Batched<B>, Action::Batched<B>, Extra::Batched<B>> {
        let batched_episode = BatchedTransition {
            observations: Obs::batch(self.observations.clone(), &self.device),
            actions: Action::batch(self.actions.clone(), &self.device),
            rewards: f32::batch(self.rewards.clone(), &self.device),
            next_observations: Obs::batch(self.next_observations.clone(), &self.device),
            terminated: f32::batch(self.terminated.clone(), &self.device),
            truncated: f32::batch(self.truncated.clone(), &self.device),
            extras: Extra::batch(self.extras.clone(), &self.device),

            batch_size: self.len(),

            _backend: PhantomData
        };

        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_observations.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.extras.clear();

        batched_episode
    }

    // 2. dynamic allocation

    // pub fn pop(mut self) -> (BatchedTransition<B, Obs::Batched, Action::Batched, Extra::Batched>, Self) {
    //     let batched_episode = BatchedTransition {
    //         observations: Obs::batch(self.observations.clone(), &self.device),
    //         actions: Action::batch(self.actions.clone(), &self.device),
    //         rewards: f32::batch(self.rewards.clone(), &self.device),
    //         next_observations: Obs::batch(self.next_observations.clone(), &self.device),
    //         terminated: f32::batch(self.terminated.clone(), &self.device),
    //         truncated: f32::batch(self.truncated.clone(), &self.device),
    //         extras: Extra::batch(self.extras.clone(), &self.device),

    //         _backend: PhantomData
    //     };

    //     (batched_episode, Self::new(self.device))
    // }

}