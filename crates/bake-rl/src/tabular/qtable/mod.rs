//! A action value (Q value) table for tabular algorithms
use crate::core::constraint::DiscreteConstraint;
use crate::exploration::DiscreteActionExploration;
use rand::prelude::*;
/// Q Table
#[derive(Debug, Clone)]
pub struct QTable {
    table: Vec<f32>,
    n_obs: usize,
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
            n_obs,
            n_action,
        } 
    }

    /// return the number of observations
    pub fn n_obs(&self) -> usize { self.n_obs }

    /// return the number of actions
    pub fn n_action(&self) -> usize { self.n_action }

    /// return a q values of given observation
    pub fn qvalues(&self, obs: usize, constraint: impl DiscreteConstraint) -> Vec<f32> {
        let qvalues = &self.table[obs * self.n_action..(obs + 1) * self.n_action];
        constraint.apply(qvalues, -1e9)
    }

    /// return a q value of given (obs, action)
    pub fn qvalue(&self, obs: usize, action: usize) -> f32 {
        self.table[obs * self.n_action + action]
    }

    /// sample an action using given exploration strategy
    pub fn action<E: DiscreteActionExploration, R: Rng>(&self, rng: &mut R, exploration: &mut E, obs: usize, constraint: impl DiscreteConstraint) -> usize {
        let qvalues = self.qvalues(obs, constraint.clone());
        exploration.sample(rng, &qvalues, constraint)
    }

    /// get a probability of choosing given action from given observation
    pub fn prob<E: DiscreteActionExploration>(&self, exploration: &mut E, obs: usize, constraint: impl DiscreteConstraint, action: usize) -> f32 {
        let qvalues = self.qvalues(obs, constraint.clone());
        exploration.prob(&qvalues, constraint, action)
    }

    /// update the inner table
    pub(crate) fn update(&mut self, obs: usize, action: usize, value: f32) {
        self.table[obs * self.n_action + action] = value;
    }
}
