//! A helper struct for taking a step in RL environments and making Transition object
use burn::Tensor;

use crate::{env::Env, types::{Batch, Batchable}};
pub struct Tape<E: Env> {
    pub obs: E::Obs,
    pub constraint: E::Constraint,
}

impl<E: Env> Tape<E> {
    pub fn new(env: &mut E) -> Self {
        let (obs, constraint) = env.reset();
        Self {
            obs,
            constraint
        }
    }

    pub fn reset(&mut self, env: &mut E) {
        let (obs, mask) = env.reset();
        self.obs = obs;
        self.constraint = mask;
    }

    pub fn step(&mut self, env: &mut E, actions: E::Action) -> (Batch<E::Obs, E::Action, E::Constraint>, f32, bool, bool) {
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
        (t, reward, terminated, truncated)
    }
}