//! Q-value tables which all tabular algorithms use
//! 

use crate::{constraint::Constraint, explore::Exploration};

/// The Q(s,a) table for tabular algorithms
#[derive(Debug)]
pub struct QTable {
    table: Vec<f32>,
    n_states: usize,
    n_actions: usize,
}

impl QTable {
    /// create a new QTable struct
    pub fn new(n_states: usize, n_actions: usize) -> Self {
        Self {
            n_states,
            n_actions,
            table: vec![0f32; n_states * n_actions]
        }
    }

    /// returns a view of q values of given state
    pub fn qvalues(&self, obs: usize) -> &[f32] {
        &self.table[obs * self.n_actions..(obs + 1) * self.n_actions]
    }

    /// return a mutable view of q values of given state
    pub fn qvalues_mut(&mut self, obs: usize) -> &mut [f32] {
        &mut self.table[obs * self.n_actions..(obs + 1) * self.n_actions]
    }

    /// return q values as `Vec<f32>`
    pub fn qvalues_as_vec(&self, obs: usize) -> Vec<f32> {
        let mut qvalues = vec![0f32; self.n_actions];
        qvalues.copy_from_slice(self.qvalues(obs));
        qvalues
    }

    /// returns an expectation of q values, according to a given policy
    pub fn expectation<C: Constraint, E: Exploration>(&self, exploration: &E, obs: usize, constraint: C) -> f32 {
        let mut expected = 0f32;
        let qvalues = self.qvalues(obs);
        for action in constraint.iter().filter_map(|(a, p)| if *p { Some(a) } else { None }) {
            expected += exploration.prob(self, obs, action, constraint) * qvalues[action];
        }
        expected
    }
    
    /// return the number of states
    pub fn n_states(&self) -> usize { self.n_states }
    /// return the number of actions
    pub fn n_actions(&self) -> usize { self.n_actions }
}

/// traits for argmax and max of qvalues
pub trait QValues {
    /// returns maximum value and the index of it
    fn max<C: Constraint>(&self, constraint: C) -> f32;
    /// returns the indices with max values
    fn argmaxes<C: Constraint>(&self, constraint: C) -> Vec<usize>;
}

impl QValues for [f32] {
    fn max<C: Constraint>(&self, constraint: C) -> f32 {
        let mut max = f32::MIN;
        for (i, &p) in constraint.iter() {
            if p {
                let v = self[i];
                if max < v { max = v; }
            }
        }
        max
    }

    fn argmaxes<C: Constraint>(&self, constraint: C) -> Vec<usize> {
        let max = self.max(constraint);
        constraint.iter().filter(|(a, p)| **p && (self[*a] - max).abs() < 1e-6).map(|(a, _)| a).collect()
    }
}