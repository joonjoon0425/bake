//! A helper struct for taking a step in RL environments and making Transition object
use crate::{env::Env, types::Transition};
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

    pub fn step(&mut self, env: &mut E, action: E::Action) -> Transition<E::Obs, E::Action, E::Constraint> {
        let ((next_obs, next_mask), reward, terminated, truncated) = env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let constraint = std::mem::replace(&mut self.constraint, next_mask);
        let t = Transition {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),
            terminated,
            truncated,
            constraint,
            next_constraints: self.constraint.clone(),
            extra: ()
        };
        t
    }
}