//! A helper struct for taking a step in RL environments and making Transition object
use crate::{env::Env, types::Transition};
pub struct Tape<E: Env> {
    pub obs: E::Obs,
    pub barrier: E::Constraint,
}

impl<E: Env> Tape<E> {
    pub fn new(env: &mut E) -> Self {
        let (obs, barrier) = env.reset();
        Self {
            obs,
            barrier
        }
    }

    pub fn reset(&mut self, env: &mut E) {
        let (obs, mask) = env.reset();
        self.obs = obs;
        self.barrier = mask;
    }

    pub fn step(&mut self, env: &mut E, action: E::Action) -> Transition<E::Obs, E::Action, E::Constraint> {
        let ((next_obs, next_mask), reward, terminated, truncated) = env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let barrier = std::mem::replace(&mut self.barrier, next_mask);
        let t = Transition {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),
            terminated,
            truncated,
            constraint: barrier,
            next_constraints: self.barrier.clone(),
            extra: ()
        };
        t
    }
}