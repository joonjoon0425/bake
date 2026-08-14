//! A helper struct for taking a step in RL env and generating a transition

use crate::{env::Env, types::{Mask, Transition}};

/// A helper struct for taking a step in RL env and generating a transition
pub struct Tape<M: Mask> {
    /// current state
    pub obs: usize,
    /// mask of current state
    pub mask: M,
}

impl<M: Mask> Tape<M> {
    /// crate new Tap object with given env. The reset() of env is called here.
    pub fn new<E: Env<Mask = M>>(env: &mut E) -> Self {
        let (obs, mask) = env.reset();
        Self {
            obs,
            mask
        }
    }

    /// reset the tape and given environment
    pub fn reset<E: Env<Mask = M>> (&mut self, env: &mut E) {
        let (obs, mask) = env.reset();
        self.obs = obs;
        self.mask = mask;
    }

    /// take one step ahead with given env and action, and returns a transition object
    pub fn step<E: Env<Mask = M>> (&mut self, env: &mut E, action: usize) -> Transition<M> {
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