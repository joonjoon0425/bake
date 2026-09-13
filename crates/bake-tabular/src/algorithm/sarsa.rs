//! An implementation of tabular Sarsa algorithm
//! 
use crate::{constraint::Constraint, data::Transition, qtable::QTable};

/// A state for Sarsa
#[derive(Debug, Clone)]
pub struct Sarsa {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

impl Sarsa {
    /// update the qtable with given state, using SARSA algorithm
    pub fn update<C: Constraint>(state: &Sarsa, qtable: &mut QTable, t: Transition<C>) {
        let target = t.reward + state.gamma * (if t.terminated { 0f32 } else { qtable.qvalues(t.next_obs)[t.get("next_action").unwrap() as usize] });
        let qvalue = qtable.qvalues(t.obs)[t.action];
        qtable.qvalues_mut(t.obs)[t.action] += state.alpha * (target - qvalue);
    }
}