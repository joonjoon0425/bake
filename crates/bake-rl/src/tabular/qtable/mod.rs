//! A action value (Q value) table for tabular algorithms
use crate::core::{constraint::NativeDiscreteConstraint, action_selector::NativeDiscreteActionSelector};
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
    pub fn qvalues(&self, obs: usize, constraint: impl NativeDiscreteConstraint) -> Vec<f32> {
        let qvalues = &self.table[obs * self.n_action..(obs + 1) * self.n_action];
        constraint.apply(qvalues, -1e9)
    }

    /// return a q value of given (obs, action)
    pub fn qvalue(&self, obs: usize, action: usize) -> f32 {
        self.table[obs * self.n_action + action]
    }

    /// sample an action using given exploration strategy
    pub fn action<A: NativeDiscreteActionSelector>(&self, action_selector: &mut A, obs: usize, constraint: impl NativeDiscreteConstraint) -> usize {
        let qvalues = self.qvalues(obs, constraint.clone());
        action_selector.sample(&qvalues, constraint)
    }

    /// update the inner table
    pub(crate) fn update(&mut self, obs: usize, action: usize, value: f32) {
        self.table[obs * self.n_action + action] = value;
    }
}
