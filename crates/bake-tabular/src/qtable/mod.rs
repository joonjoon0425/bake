//! Q-value tables which all tabular algorithms use
//! 

use crate::constraint::Constraint;

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
        let mut max = self[0];
        for (i, &p) in constraint.iter() {
            if p {
                let v = self[i];
                if max < v { max = v; }
            }
        }
        max
    }

    fn argmaxes<C: Constraint>(&self, constraint: C) -> Vec<usize> {
        let mut max = self[0];
        let mut indices = vec![0usize];
        for (i, &p) in constraint.iter().skip(1) {
            if p {
                let v = self[i];
                if max < v { max = v; indices.clear(); indices.push(i) }
                else if (max - v).abs() < 1e-6 { indices.push(i) }
            }
        }
        indices
    }
}