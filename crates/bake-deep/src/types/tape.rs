//! A helper struct for taking a step in RL environments and making Transition object
use burn::Tensor;

use crate::{env::Env, types::{Batch, Batchable}};
pub struct Tape<E: Env> {
    pub obs: E::Obs,
    pub constraint: E::Constraint,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

impl<E: Env> Tape<E> {
    pub fn new(env: &mut E) -> Self {
        let (obs, constraint) = env.reset();
        Self {
            obs,
            constraint,
            reward: 0f32,
            terminated: false,
            truncated: false,
        }
    }

    pub fn reset(&mut self, env: &mut E) {
        let (obs, mask) = env.reset();
        self.obs = obs;
        self.constraint = mask;
    }

    pub fn step(&mut self, env: &mut E, actions: E::Action) -> Batch<E::Obs, E::Action, E::Constraint> {
        let ((next_obs, next_mask), reward, terminated, truncated) = env.step(actions.clone());
        let obss = std::mem::replace(&mut self.obs, next_obs);
        let device = obss.device();
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
            extras: ()
        };
        self.reward = reward;
        self.terminated = terminated;
        self.truncated = truncated;
        t
    }

    pub fn done(&self) -> bool { self.terminated || self.truncated }
}