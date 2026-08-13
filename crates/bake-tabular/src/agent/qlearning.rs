//! Q-Learning algorithm Implementation
use crate::{policy::{EpsGreedy, max}, types::{Mask, Step}};

/// An implementation of Q-Learning algorithm
pub struct QLearningAgent {
    gamma: f32,
    alpha: f32,

    q_table: Vec<f32>,

    n_actions: usize,
}

impl QLearningAgent {
    /// Create a new QLearningAgent
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            gamma,
            alpha,
            q_table: vec![0f32; n_states * n_actions],
            n_actions,
        }
    }

    /// Choose an action, using the given policy
    pub fn action<M: Mask>(&self, policy: &mut EpsGreedy, obs: usize, mask: M) -> usize {
        let start = obs * self.n_actions;
        policy.sample(&self.q_table[start..start + self.n_actions], mask)
    }

    /// Update the QTable according to given transition
    pub fn update<M: Mask>(&mut self, obs: usize, action: usize, next_mask: M, step: Step) {
        let start = step.obs * self.n_actions;
        let next_qvalue = max(&self.q_table[start..start + self.n_actions], next_mask);
        let qvalues = &mut self.q_table[obs * self.n_actions + action];
        let target = step.reward + if step.done { 0f32 } else { self.gamma * next_qvalue };
        *qvalues += self.alpha * (target - *qvalues);
    }
}