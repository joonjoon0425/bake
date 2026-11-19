//! n-step SARSA
use crate::core::{constraint::DiscreteConstraint, data::Transition};
use crate::tabular::qtable::QTable;

/// state struct for n-step SARSA
#[derive(Debug, Clone)]
pub struct NStepSARSAState {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

/// update q table using n-step SARSA algorithm
/// # Note
/// - requires following `key: element type` in transitions
///     - `next_action: usize`
pub fn n_step_sarsa_update(state: &NStepSARSAState, qtable: &mut QTable, transitions: Vec<Transition<usize, impl DiscreteConstraint, usize, f32>>) {
    let first = transitions.first().unwrap().clone();
    let mut last = transitions.last().unwrap().clone();
    let n = transitions.len();
    let mut g = 0f32;
    for t in transitions.into_iter().rev() {
        g = state.gamma * g + t.reward
    }

    if !last.terminated { g += state.gamma.powi(n as i32) * qtable.qvalue(last.next_obs, last.dict.remove_str("next_action").expect("`n_step_sarsa_update` expects key `next_action` with type `usize`")); }
    let q_value = qtable.qvalue(first.obs, first.action);
    qtable.update(first.obs, first.action, q_value + state.alpha * (g - q_value)); 
}