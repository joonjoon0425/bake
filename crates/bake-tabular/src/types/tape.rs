//! A helper struct for taking a step in RL env and generating a transition

use crate::{env::Env, types::{Mask, Transition}};

/// A helper struct for taking a step in RL env and generating a transition
pub struct Tape<M: Mask> {
    obs: usize,
    mask: M,
}

impl<M: Mask> Tape<M> {
    /// crate new Tap object with given obs and mask
    pub fn new(obs: usize, mask: M) -> Self {
        Self {
            obs,
            mask
        }
    }

    /// take one step ahead with given env and action, and returns a transition object
    pub fn step<E: Env<Mask = M>, Extra> (&mut self, env: &mut E, action: usize) -> Transition<M> {
        let (next_obs, reward, terminated, truncated, next_mask) = env.step(action);
        let t = Transition {
            obs: self.obs,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
            mask: self.mask,
            next_mask,
            extra: ()
        };

        self.obs = next_obs;
        self.mask = next_mask;

        t
    }
}