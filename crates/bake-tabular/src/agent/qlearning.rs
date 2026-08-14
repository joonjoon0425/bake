//! Q-Learning algorithm Implementation
use crate::{policy::{EpsGreedy}, qtable::QTable, types::{Mask, Transition}};

/// An implementation of Q-Learning algorithm
pub struct QLearningAgent {
    gamma: f32,
    alpha: f32,

    qtable: QTable,
}

impl QLearningAgent {
    /// Create a new QLearningAgent
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            gamma,
            alpha,
            qtable: QTable::new(n_states, n_actions),
        }
    }

    /// Choose an action, using the given policy
    pub fn action<M: Mask>(&self, policy: &mut EpsGreedy, obs: usize, mask: M) -> usize {
        policy.sample(self.qtable.row(obs), mask)
    }

    /// Update the QTable according to given transition
    pub fn update<M: Mask>(&mut self, t: Transition<M>) {
        let target = t.reward + if t.terminated { 0f32 } else { self.gamma * self.qtable.max(t.next_obs, t.next_mask) };
        let qvalues = &mut self.qtable[(t.obs, t.action)];
        *qvalues += self.alpha * (target - *qvalues);
    }
}