//! Q-Learning
use crate::core::{constraint::DiscreteConstraint, data::NativeTransition};
use crate::tabular::qtable::QTable;

/// state struct for QLearning
#[derive(Debug, Clone)]
pub struct QLearningState {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

/// update q table using Q-Learning algorithm
pub fn qlearning_update(state: &QLearningState, qtable: &mut QTable, t: NativeTransition<usize, usize, impl DiscreteConstraint, f32>) {
    let target = t.reward + if t.terminated { 0f32 } else { state.gamma * super::argmax_max(&qtable.qvalues(t.next_obs, t.next_constraint)).1 };
    let qvalue = qtable.qvalue(t.obs, t.action);
    qtable.update(t.obs, t.action, state.alpha * (target - qvalue) + qvalue);
}