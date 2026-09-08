//! A helper struct which creates and saves the transition (s, a, r, s')
//! 
use std::collections::HashMap;

use crate::{data::Transition, env::Environment};

/// A helper struct which helps creating and taking a step in training loop
pub struct Tape<E: Environment> {
    /// environment
    env: E,
    /// current observation
    pub obs: usize,
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
    /// create new `Tape` object with given env.
    /// # Warning
    /// - the given `env`'s `return` is called once internally
    pub fn new(mut env: E) -> Self {
        let (obs, constraint) = env.reset();
        Self {
            env,
            obs,
            constraint,
            reward: 0.,
            terminated: false,
            truncated: false,
            episode_reward: 0.,
            steps: 0,
        }
    }

    /// take a step in environment with given action and return the transition object
    /// after the step, `Tape` updates reward, terminated, and truncated
    pub fn step(&mut self, action: usize) -> Transition<E::Constraint> {
        let ((next_obs, next_constraint), reward, terminated, truncated) = self.env.step(action);
        let t = Transition {
            obs: self.obs,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
            constraint: self.constraint,
            next_constraint,
            extra: HashMap::new()
        };
        self.obs = next_obs;
        self.constraint = next_constraint;
        self.reward = reward;
        self.terminated = terminated;
        self.truncated = truncated;

        self.episode_reward += reward;
        self.steps += 1;

        t
    }

    /// reset the tape and given environment
    pub fn reset(&mut self) {
        let (obs, constraint) = self.env.reset();
        self.obs = obs;
        self.constraint = constraint;
        self.reward = 0.;
        self.terminated = false;
        self.truncated = false;
        self.episode_reward = 0.;
        self.steps = 0;
    }

    /// returns true if the environment has terminated or truncated
    pub fn done(&self) -> bool { self.terminated || self.truncated }
}