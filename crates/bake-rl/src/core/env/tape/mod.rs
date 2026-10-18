//! The `Tape` struct, which helps making transition tuple (s, a, r, s') and manage environment steps and resets.
//! - Especially useful for vector environments
//! 
use crate::core::{data::Batch, env::{VectorEnvironment, space::Space}};
use burn::prelude::*;
/// The `Tape` struct, which helps making transition tuple (s, a, r, s') and manage environment steps and resets.
/// - Especially useful for vector environments
pub struct Tape<E: VectorEnvironment> {
    env: E,
    /// current observation
    pub obs: <E::ObsSpace as Space>::Element,
    /// current constraint
    pub constraint: E::Constraint,
    /// `true` if the environment has terminated or truncated
    done: Tensor<1, Bool>,
    /// current information of the environment
    pub info: E::Info,
    autoreset: bool,
}

impl<E: VectorEnvironment> Tape<E> {
    /// create a new tape, calling `reset` on all environments
    /// # Arguments
    /// - `env: E`
    ///     - vector environment which the tape will manage
    /// - `autoreset: bool`
    ///     - `false`: turns off the autoreset mode
    ///     - `true`: turns on the autoreset mode
    ///         - automatically resets the environments which are terminated or truncated
    pub fn new(mut env: E, autoreset: bool) -> Self {
        let ((obs, constraint), info) = env.reset(None);
        let n_envs = env.n_envs();
        let device = env.device();
        Self {
            env,
            obs,
            constraint,
            done: Tensor::from_bool(vec![false; n_envs].as_slice(), &device),
            info,
            autoreset,
        }
    }

    /// reset the environments, according to given mask. If `None` was given, resets all environments
    /// - 0: does not reset
    /// - 1: reset
    pub fn reset(&mut self, mask: Option<Tensor<1, Bool>>) {
        let ((obs, constraint), info) = self.env.reset(mask);
        self.obs = obs;
        self.constraint = constraint;
        self.done = Tensor::zeros_like(&self.done);
        self.info = info;
    }
    /// take one step and returns the transition and info
    /// - if `autoreset = true`, during the `step`, terminated or truncated environments are automatically reset
    pub fn step(&mut self, action: <E::ActionSpace as Space>::Element) -> Batch<<E::ObsSpace as Space>::Element, <E::ActionSpace as Space>::Element, E::Constraint, E::Reward> {
        let ((next_obs, next_constraint), reward, terminated, truncated, info) = self.env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let constraint = std::mem::replace(&mut self.constraint, next_constraint);
        let done = terminated.clone().bool_or(truncated.clone());
        self.done = done.clone();
        self.info = info;
        let batch = Batch {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),

            constraint,
            next_constraint: self.constraint.clone(),

            terminated,
            truncated
        };
        
        if self.autoreset { self.reset(Some(done)) };

        batch
    }
    /// returns the mask, which is true if the environment has terminated or truncated
    pub fn done(&self) -> &Tensor<1, Bool> {
        &self.done
    }
}