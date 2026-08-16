//! A helper struct for taking a step in RL environments and making Transition object
use crate::{env::Env, types::Transition};
pub struct Tape<E: Env> {
    pub obs: E::Obs,
    pub mask: E::Mask,
}

impl<E: Env> Tape<E> {
    pub fn new(env: &mut E) -> Self {
        let (obs, mask) = env.reset();
        Self {
            obs,
            mask
        }
    }

    pub fn reset(&mut self, env: &mut E) {
        let (obs, mask) = env.reset();
        self.obs = obs;
        self.mask = mask;
    }

    pub fn step(&mut self, env: &mut E, action: E::Action) -> Transition<E::Obs, E::Action, E::Mask> {
        let ((next_obs, next_mask), reward, terminated, truncated) = env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let mask = std::mem::replace(&mut self.mask, next_mask);
        let t = Transition {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),
            terminated,
            truncated,
            mask,
            next_mask: self.mask.clone(),
            extra: ()
        };
        t
    }
}