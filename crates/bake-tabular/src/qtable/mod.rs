//! # The Q-table struct for tabular algorithms

use std::ops::{Index, IndexMut};

use crate::types::Mask;

/// A Q-table struct
pub struct QTable {
    table: Vec<f32>,
    n_states: usize,
    n_actions: usize,
}

impl QTable {
    /// Create a new QTable struct, using given number of states and actions
    pub fn new(n_states: usize, n_actions: usize) -> Self {
        Self {
            n_states,
            n_actions,
            table: vec![0f32; n_states * n_actions]
        }
    }

    /// returns a view of q values of given state
    pub fn row(&self, obs: usize) -> &[f32] {
        &self.table[obs * self.n_actions..(obs + 1) * self.n_actions]
    }

    /// returns a mutable view of q values of given state
    pub fn row_mut(&mut self, obs: usize) -> &mut [f32] {
        &mut self.table[obs * self.n_actions..(obs + 1) * self.n_actions]
    }

    /// compute the maximum q values along the q values of corresponding state
    pub fn max<M: Mask>(&self, obs: usize, mask: M) -> f32 {
        let mut qmax = f32::MIN;
        let qvalues = self.row(obs);
        for i in mask.possible_actions() {
            if qvalues[i] > qmax {
                qmax = qvalues[i];
            }
        }
        qmax
    }

    /// returns a vector of actions with maximum q values
    pub fn argmaxes<M: Mask>(&self, obs: usize, mask: M) -> Vec<bool> {
        let mut qmax = f32::MIN;
        let qvalues = self.row(obs);
        let mut candidates = vec![false; mask.n_actions()];
        for i in mask.possible_actions() {
            if qvalues[i] > qmax {
                candidates.fill(false);
                candidates[i] = true;
                qmax = qvalues[i];
            } else if qvalues[i] - qmax < 1e-10 {
                candidates[i] = true;
            }
        }
        candidates
    }

    /// returns the number of states
    pub fn n_states(&self) -> usize { self.n_states }
    /// return the number of actions
    pub fn n_actions(&self) -> usize { self.n_actions }
}

impl Index<(usize, usize)> for QTable {
    type Output = f32;

    fn index(&self, (obs, action): (usize, usize)) -> &Self::Output {
        &self.table[obs * self.n_actions + action]
    }
}

impl IndexMut<(usize, usize)> for QTable {
    fn index_mut(&mut self, (obs, action): (usize, usize)) -> &mut Self::Output {
        &mut self.table[obs * self.n_actions + action]
    }
}