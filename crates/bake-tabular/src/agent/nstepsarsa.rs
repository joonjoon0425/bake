//! N-Step Sarsa algorithm implementation

use crate::{policy::Policy, qtable::QTable, types::{Mask, Transition}};

/// An implementation of n step sarsa algorithm
pub struct NStepSarsaAgent {
    gamma: f32,
    alpha: f32,

    qtable: QTable,
}

impl NStepSarsaAgent {
    /// Create a new `NStepSarsaAgent`
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
    pub fn update<M: Mask>(&mut self, t: &[Transition<M, usize>]) {
        let last_t = t.last().unwrap();
        let mut target = if last_t.terminated { 0f32 } else { self.qtable[(last_t.next_obs, last_t.extra)] };
        for transition in t.iter().rev() {
            target = self.gamma * target + transition.reward;
        }
        let first_t = t.first().unwrap();
        let qvalues = &mut self.qtable[(first_t.obs, first_t.action)];
        *qvalues += self.alpha * (target - *qvalues);
    }
}