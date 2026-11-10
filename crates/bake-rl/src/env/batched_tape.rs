use crate::core::data::{Shaped, BatchedDict, BatchedTransition};
use crate::core::env::{BatchedEnvironment, space::BatchedSpace};
use burn::prelude::*;

/// The `BatchedTape` struct, which helps making transition tuple (s, a, r, s') and manage batched single and vector environment steps and resets.
pub struct BatchedTape<E: BatchedEnvironment> {
    env: E,
    /// current observation
    pub obs: <E::ObsSpace as BatchedSpace>::Element,
    /// current constraint
    pub constraint: E::Constraint,
    /// current information of the environment
    pub info: E::Info,
    /// `true` if the environment has terminated or truncated
    done: Tensor<1, Bool>,
    autoreset: bool,
}

impl<E: BatchedEnvironment> BatchedTape<E> {
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
    /// - if `autoreset = true`
    ///     - during the `step`, terminated or truncated environments are automatically reset
    ///     - the returned `info` is the information before being automatically reset; if you read `info` from directly like `tape.info`, you will get the reset-info
    pub fn step(&mut self, action: <E::ActionSpace as BatchedSpace>::Element) -> (Shaped<BatchedTransition<<E::ObsSpace as BatchedSpace>::Element, E::Constraint, <E::ActionSpace as BatchedSpace>::Element, E::Reward>>, E::Info) {
        let ((next_obs, next_constraint), reward, terminated, truncated, info) = self.env.step(action.clone());
        let obs = std::mem::replace(&mut self.obs, next_obs);
        let constraint = std::mem::replace(&mut self.constraint, next_constraint);
        let done = terminated.clone().bool_or(truncated.clone());
        
        let bs = vec![self.env.n_envs()];

        self.done = done.clone();
        self.info = info.clone();
        let batch = BatchedTransition {
            obs,
            action,
            reward,
            next_obs: self.obs.clone(),

            constraint,
            next_constraint: self.constraint.clone(),

            terminated,
            truncated,
            dict: BatchedDict::new()
        };
        
        if self.autoreset {
            // later, change this so that the synchronized vector environment holds an already-batched observation and constraints
            self.reset(Some(done))
        };

        (Shaped::new(batch, bs), info)
    }
    
    /// returns the mask, which is true if the environment has terminated or truncated
    pub fn done(&self) -> Tensor<1, Bool> {
        self.done.clone()
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