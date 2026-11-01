//! A action value (Q value) table for tabular algorithms
use burn::prelude::*;

use crate::core::{constraint::DiscreteConstraint, exploration::Exploration};
/// Q Table
#[derive(Debug, Clone)]
pub struct QTable {
    table: Vec<f32>,
    n_action: usize,
}

impl QTable {
    /// create a new `QTable`
    /// # Arguments
    /// - n_obs: the number of observations
    /// - n_action: the number of actions
    pub fn new(n_obs: usize, n_action: usize) -> Self {
        let table = vec![0.; n_obs * n_action];
        Self {
            table,
            n_action,
        } 
    }

    /// return a q values of given observation
    /// # Warning
    /// - note that the first axis will always be 1
    pub fn qvalues(&self, obs: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let device = obs.device();
        let obs = obs.into_scalar::<u64>() as usize;
        let start = obs * self.n_action;
        let qvalues = self.table[start..start + self.n_action].to_vec();
        let qvalues = Tensor::<1>::from_floats(qvalues.as_slice(), &device).unsqueeze_dim(0);
        constraint.apply(qvalues, -1e9)
    }

    /// return a q value of given (obs, action)
    pub fn qvalue(&self, obs: usize, action: usize) -> f32 {
        self.table[obs * self.n_action + action]
    }

    /// sample an action using given exploration strategy
    pub fn action<E: Exploration>(&self, exploration: &mut E, obs: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = self.qvalues(obs, constraint.clone());
        exploration.sample(qvalues, constraint)
    }

    /// update the inner table
    pub(crate) fn update(&mut self, obs: usize, action: usize, value: f32) {
        self.table[obs * self.n_action + action] = value;
    }
}
