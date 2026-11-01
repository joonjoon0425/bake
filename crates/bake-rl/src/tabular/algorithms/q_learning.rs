//! Q-Learning
use burn::prelude::*;
use crate::core::{constraint::DiscreteConstraint, data::Transition};
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
pub fn qlearning_update(state: &QLearningState, qtable: &mut QTable, t: Transition<Tensor<1, Int>, Tensor<1, Int>, impl DiscreteConstraint, Tensor<1>>) {
    let obs = t.obs.into_scalar::<u64>() as usize;
    let action = t.action.into_scalar::<u64>() as usize;
    let target = (t.reward + state.gamma * qtable.qvalues(t.next_obs, t.next_constraint).max_dim(1).squeeze_dim(0) * t.terminated.bool_not().float()).into_scalar::<f32>();
    let qvalue = qtable.qvalue(obs, action);
    qtable.update(obs, action, state.alpha * (target - qvalue) + qvalue);
}