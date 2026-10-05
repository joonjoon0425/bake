//! An implementation of tabular Q-learning algorithm
//! 
use crate::{constraint::Constraint, data::Transition, qtable::{QTable, QValues}};

/// A state for Q-learning
#[derive(Debug, Clone)]
pub struct QLearning {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

impl QLearning {
    /// update the qtable with given state, using Q-Learning algorithm
    pub fn update<C: Constraint>(state: &QLearning, qtable: &mut QTable, t: Transition<C>) {
        let target = t.reward + state.gamma * (if t.terminated { 0f32 } else { qtable.qvalues(t.next_obs).max(t.next_constraint) });
        let qvalue = qtable.qvalues(t.obs)[t.action];
        qtable.qvalues_mut(t.obs)[t.action] += state.alpha * (target - qvalue);
    }
}