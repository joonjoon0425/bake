//! A helper struct which creates and saves the transition (s, a, r, s')
//! 
use burn::Tensor;

use crate::{data::{Batch, extras::ExtraContainer}, env::Environment};

/// A helper struct which helps creating and taking a step in training loop
pub struct Tape<E: Environment> {
    /// environment
    env: E,
    /// current observation
    pub obs: E::Obs,
    /// current constraint
    pub constraint: E::Constraint,
    /// next reward
    pub reward: f32,
    /// if next observation is in terminal state, true
    pub terminated: bool,
    /// if the environment has truncated, true
    pub truncated: bool,

    /// cummulative episodic reward
    pub episode_reward: f32,
    /// cummulative episodic steps
    pub steps: usize,
}

impl<E: Environment> Tape<E> {
    /// create a new tape struct
    /// # Warning
    /// calls `reset` on given environment
    pub fn new(mut env: E) -> Self {
        let (obs, constraint) = env.reset();
        Self {
            env,
            obs,
            constraint,
            reward: 0f32,
            terminated: false,
            truncated: false,
            episode_reward: 0f32,
            steps: 0usize,
        }
    }

    /// reset the environment and itself
    pub fn reset(&mut self) {
        let (obs, mask) = self.env.reset();
        self.obs = obs;
        self.constraint = mask;

        self.reward = 0f32;
        self.terminated = false;
        self.truncated = false;
        self.episode_reward = 0f32;
        self.steps = 0usize;
    }

    /// take a step in environment with given action and return the transition object
    /// after the step, `Tape` updates reward, terminated, and truncated
    pub fn step(&mut self, actions: E::Action) -> Batch<E::Obs, E::Action, E::Constraint> {
        let device = self.env.device();

        let ((next_obs, next_mask), reward, terminated, truncated) = self.env.step(actions.clone());
        let obss = std::mem::replace(&mut self.obs, next_obs);
        let constraints = std::mem::replace(&mut self.constraint, next_mask);
        let t = Batch {
            obss,
            actions,
            rewards: Tensor::from_floats([reward], &device),
            next_obss: self.obs.clone(),
            terminated: Tensor::from_floats([if terminated { 1f32 } else { 0f32 }], &device),
            truncated: Tensor::from_floats([if truncated { 1f32 } else { 0f32 }], &device),
            constraints,
            next_constraints: self.constraint.clone(),
            extras: ExtraContainer::new(),
        };
        self.reward = reward;
        self.terminated = terminated;
        self.truncated = truncated;
        
        self.episode_reward += reward;
        self.steps += 1;
        t
    }

    /// returns true if the environment has terminated or truncated
    pub fn done(&self) -> bool { self.terminated || self.truncated }
}