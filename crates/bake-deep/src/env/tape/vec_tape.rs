//! A `Tape` struct for Vectorized Environment
use crate::{data::{Batch, extras::ExtraContainer}, env::vec::VectorizedEnvironment};
use burn::prelude::*;
/// A `VecTape` struct for vectorized environment
pub struct VecTape<Ve: VectorizedEnvironment> {
    // environments
    envs: Ve,
    /// current observations
    pub obss: Ve::Obs,
    /// current constraints
    pub constraints: Ve::Constraint,
    /// current reward
    pub rewards: Tensor<1>,
    /// if next observation is in terminal state, true
    pub terminated: Tensor<1>,
    /// if the environment has truncated, true
    pub truncated: Tensor<1>,

    /// cummulative episode reward
    pub episode_rewards: Tensor<1>,
    /// cummulative episodic steps
    pub steps: Tensor<1>,
}

impl<Ve: VectorizedEnvironment> VecTape<Ve> {
    /// create a new `VecTape` struct
    /// # Warning
    /// calls 'reset' on given environments
    pub fn new(mut envs: Ve) -> Self {
        let n_envs = envs.n_envs();
        let device = envs.device();
        let (obss, constraints) = envs.reset_all();
        let rewards = Tensor::zeros([n_envs], &device);
        let terminated = Tensor::zeros([n_envs], &device);
        let truncated = Tensor::zeros([n_envs], &device);
        let episode_rewards = Tensor::zeros([n_envs], &device);
        let steps = Tensor::zeros([n_envs], &device);
        Self {
            envs,
            obss,
            constraints,
            rewards,
            terminated,
            truncated,
            episode_rewards,
            steps
        }
    }

    /// take a step in environment with given action and return the transition object
    /// after the step, `VecTape` updates reward, terminated, and truncated
    pub fn step(&mut self, actions: Ve::Action) -> Batch<Ve::Obs, Ve::Action, Ve::Constraint> {
        let mask: Tensor<1> = (1 - self.terminated.clone()) * (1 - self.truncated.clone());
        self.episode_rewards.inplace(|r| r * mask.clone());
        self.steps.inplace(|r| r * mask);
        let ((next_obss, next_constraints), rewards, terminated, truncated, (final_obss, final_constraints)) = self.envs.step(actions.clone());
        let obss = std::mem::replace(&mut self.obss, next_obss);
        let constraints = std::mem::replace(&mut self.constraints, next_constraints);

        let t = Batch {
            obss,
            constraints,
            actions,
            next_obss: final_obss,
            next_constraints: final_constraints,
            rewards: rewards.clone(),
            terminated: terminated.clone(),
            truncated: truncated.clone(),
            extras: ExtraContainer::new(),
        };
        self.rewards = rewards.clone();
        self.terminated = terminated;
        self.truncated = truncated;
        
        self.episode_rewards.inplace(|r| r + rewards);
        self.steps.inplace(|s| s + Tensor::ones([self.envs.n_envs()], &self.envs.device()));
        t
    }

    /// return the tensor mask of terminated or truncated environments
    pub fn done(&self) -> Tensor<1> { 1 - (1 - self.terminated.clone()) * (1 - self.truncated.clone()) }
}