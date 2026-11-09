
use crate::core::data::Transition;
use crate::env::{Environment, space::Space};
/// The `Tape` struct, which helps making transition tuple (s, a, r, s') and manage vector environment steps and resets.
pub struct Tape<E: Environment> {
    env: E,
    /// current observation
    pub obs: <E::ObsSpace as Space>::Element,
    /// current constraint
    pub constraint: E::Constraint,
    /// current information of the environment
    pub info: E::Info,
    /// `true` if the environment has terminated or truncated
    done: bool,
}

impl<E: Environment> Tape<E> {
    /// create a new tape, calling `reset` on all environments
    /// # Arguments
    /// - `env: E`
    ///     - vector environment which the tape will manage
    pub fn new(mut env: E) -> Self {
        let ((obs, constraint), info) = env.reset();
        Self {
            env,
            obs,
            constraint,
            done: false,
            info,
        }
    }

    /// reset the environment
    pub fn reset(&mut self) {
        let ((obs, constraint), info) = self.env.reset();
        self.obs = obs;
        self.constraint = constraint;
        self.done = false;
        self.info = info;
    }
    
    /// take one step and returns the transition and info
    pub fn step(&mut self, action: <E::ActionSpace as Space>::Element) -> (Transition<<E::ObsSpace as Space>::Element, E::Constraint, <E::ActionSpace as Space>::Element, E::Reward>, E::Info) {
        let ((next_obs, next_constraint), reward, terminated, truncated, info) = self.env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let constraint = std::mem::replace(&mut self.constraint, next_constraint);
        let done = terminated || truncated;

        self.done = done;
        self.info = info.clone();
        let transition = Transition {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),

            constraint,
            next_constraint: self.constraint.clone(),

            terminated,
            truncated,
            dict: (),
        };

        (transition, info)
    }
    /// returns the mask, which is true if the environment has terminated or truncated
    pub fn done(&self) -> bool {
        self.done
    }
    /// returns the observation space
    pub fn obs_space(&self) -> &E::ObsSpace {
        self.env.obs_space()
    }
    /// returns the action space
    pub fn action_space(&self) -> &E::ActionSpace {
        self.env.action_space()
    }
}