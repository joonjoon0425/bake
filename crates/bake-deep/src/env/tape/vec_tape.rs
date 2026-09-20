//! A `Tape` struct for Vectorized Environment

use crate::env::Environment;
use crate::env::vec::VectorizedEnvironment;

/// A `VecTape` struct for vectorized environment
pub struct VecTape<E: Environment, Ve: VectorizedEnvironment<E>> {
    // environments
    envs: Ve,
    /// vector of current observations
    obss: Vec<E::Obs>,
    constraints: Vec<E::Constraint>,
    /// current reward
    pub rewards: Vec<f32>,
    /// if next observation is in terminal state, true
    pub terminated: Vec<bool>,
    /// if the environment has truncated, true
    pub truncated: Vec<bool>,

    /// cummulative episode reward
    pub episode_rewards: Vec<f32>,
    /// cummulative episodic steps
    pub steps: Vec<usize>,
}

impl<E: Environment, Ve: VectorizedEnvironment<E>> VecTape<E, Ve> {
    /// create a new `VecTape` struct
    /// # Warning
    /// calls 'reset' on given environments
    pub fn new(mut envs: Ve) -> Self {
        let n_envs = envs.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);

        for i in 0..n_envs {
            let (obs, constraint) = envs.reset(i);
            obss.push(obs);
            constraints.push(constraint);
        }

        Self {
            envs,
            obss,
            constraints,
            rewards: vec![0f32; n_envs],
            terminated: vec![false; n_envs],
            truncated: vec![false; n_envs],
            episode_rewards: vec![0f32; n_envs],
            steps: vec![0; n_envs]
        }
    }

    /// if the environment has terminated or truncated, reset the environment automatically
    pub fn autoreset(&mut self) {
        let n_envs = self.envs.n_envs();
        for i in 0..n_envs {
            if self.terminated[i] || self.truncated[i] { 
                let (obs, constraint) = self.envs.reset(i);
                self.obss[i] = obs;
                self.constraints[i] = constraint;
                self.rewards[i]= 0f32;
                self.terminated[i] = false;
                self.truncated[i] = false;
                self.episode_rewards[i] = 0f32;
                self.steps[i] = 0usize;
            }
        }
    }

    /// reset the environment of given index
    pub fn reset(&mut self, index: usize) {
        let (obs, constraint) = self.envs.reset(index);
        self.obss[index] = obs;
        self.constraints[index] = constraint;
        self.rewards[index]= 0f32;
        self.terminated[index] = false;
        self.truncated[index] = false;
        self.episode_rewards[index] = 0f32;
        self.steps[index] = 0usize;
    }

    /// take a step in environment with given action and return the transition object
    /// after the step, `VecTape` updates reward, terminated, and truncated
    pub fn step(&mut self, actions: E::Action) -> Batch<E::Obs, E::Action, E::Constraint> {
        let device = self.envs.device();
        let n_envs = self.envs.n_envs();
        for 
    }
}