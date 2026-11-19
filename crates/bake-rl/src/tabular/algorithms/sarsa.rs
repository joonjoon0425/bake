//! SARSA
use crate::core::{constraint::DiscreteConstraint, data::Transition};
use crate::tabular::qtable::QTable;

/// state struct for SARSA
#[derive(Debug, Clone)]
pub struct SARSAState {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

/// update q table using SARSA algorithm
/// # Note
/// - requires following `key: element type` in transitions
///     - `next_action: usize`
pub fn sarsa_update(state: &SARSAState, qtable: &mut QTable, mut t: Transition<usize, impl DiscreteConstraint, usize, f32>) {
    let target = t.reward + if t.terminated { 0f32 } else { state.gamma * qtable.qvalue(t.next_obs, t.dict.remove_str("next_action").expect("`sarsa_update` expects key `next_action` with type `usize`")) };
    let qvalue = qtable.qvalue(t.obs, t.action);
    qtable.update(t.obs, t.action, state.alpha * (target - qvalue) + qvalue);
}