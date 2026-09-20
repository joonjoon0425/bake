//! A `Tape` struct for Vectorized Environment

use crate::env::Environment;
use crate::env::vec::VectorizedEnvironment;

/// A `VecTape` struct for vectorized environment
pub struct VecTape<E: Environment, Ve: VectorizedEnvironment<E>> {
    // environments
    envs: Ve,
    /// vector of current observations
    obs: Vec<E::Obs>,
    constraint: Vec<E::Constraint>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,

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
    }
}