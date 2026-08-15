//! Expected Sarsa algorithm implementation

use crate::{policy::{EpsGreedy, Policy}, qtable::QTable, types::{Mask, Transition}};

/// An implementation of Expected Sarsa algorithm
pub struct ExpectedSarsaAgent {
    gamma: f32,
    alpha: f32,

    qtable: QTable,
}

impl ExpectedSarsaAgent {
    /// Create a new ExpectedSarsaAgent
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            gamma,
            alpha,
            qtable: QTable::new(n_states, n_actions),
        }
    }

    /// Choose an action, using the given policy
    pub fn action<M: Mask, P: Policy>(&self, policy: &mut P, obs: usize, mask: M) -> usize {
        policy.sample(self.qtable.row(obs), mask)
    }

    /// Update the QTable according to given transition and current policy
    pub fn update<M: Mask, P: Policy>(&mut self, t: Transition<M>, policy: &P) {
        let mut expected: f32 = 0f32;
        let next_qvalues = self.qtable.row(t.next_obs);
        for action in t.next_mask.possible_actions() {
            expected += policy.prob(next_qvalues, action, t.next_mask) * self.qtable[(t.next_obs, action)];
        }

        let target = t.reward + if t.terminated { 0f32 } else { self.gamma * expected };
        let qvalues = &mut self.qtable[(t.obs, t.action)];
        *qvalues += self.alpha * (target - *qvalues);
    }
}