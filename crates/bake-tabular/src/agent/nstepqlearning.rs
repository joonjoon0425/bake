//! Q-Learning algorithm Implementation
use crate::{policy::{EpsGreedy, Greedy, Policy}, qtable::QTable, types::{Mask, Transition}};

/// An implementation of n-step Q-Learning algorithm, with tree backup method
pub struct NStepQLearningAgent {
    gamma: f32,
    alpha: f32,

    qtable: QTable,
}

impl NStepQLearningAgent {
    /// Create a new NStepQLearningAgent
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

    /// Update the QTable according to given transition
    pub fn update<M: Mask>(&mut self, t: &[Transition<M>]) {
        let last_t = t.last().unwrap();
        let policy = Greedy;
        
        let mut target = last_t.reward
            + if last_t.terminated {
                0f32
            } else {
                self.gamma * self.qtable.expectation(&policy, last_t.next_obs, last_t.next_mask)
            };

        for (i, transition) in t.iter().enumerate().skip(1).rev() {
            let qvalue = self.qtable.row(transition.obs);
            let obs = transition.obs;
            let mask = transition.mask;
            let action = transition.action;
            target = t[i - 1].reward + self.gamma * (self.qtable.expectation(&policy, obs, mask) + policy.prob(qvalue, action, mask) * (target - qvalue[action]))
        }

        let first_t = t.first().unwrap();
        let qvalue = &mut self.qtable[(first_t.obs, first_t.action)];
        *qvalue += self.alpha * (target - *qvalue);
    }
}