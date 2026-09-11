//! An implementation of tabular Sarsa algorithm
//! 
use crate::{algorithm::nstep_estimator::{self, Bootstrap}, constraint::Constraint, data::Transition, explore::Exploration, qtable::QTable};

/// A state for Sarsa
#[derive(Debug, Clone)]
pub struct NStepSarsa {
    /// n
    pub n: usize,
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

impl NStepSarsa {
    /// update the qtable with given state, using SARSA algorithm
    pub fn update<C: Constraint>(state: &NStepSarsa, qtable: &mut QTable, exploration: &impl Exploration, t: Vec<Transition<C>>) {
        let target = nstep_estimator::base(qtable, exploration, state.gamma, Bootstrap::NextAction, t.clone());
        let first = t.first().unwrap();
        let qvalue = qtable.qvalues(first.obs)[first.action];
        qtable.qvalues_mut(first.obs)[first.action] += state.alpha * (target - qvalue);
    }
}