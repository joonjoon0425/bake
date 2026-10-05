//! An implementation of tabular Sarsa algorithm
//! 
use crate::{algorithm::nstep_estimator, constraint::Constraint, data::Transition, qtable::QTable};

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
    pub fn update<C: Constraint>(state: &NStepSarsa, qtable: &mut QTable, t: Vec<Transition<C>>) {
        let bootstrap = qtable.qvalues(t.last().unwrap().next_obs)[t.last().unwrap().get("next_action").expect("NStepSarsa requires the next_action attribute") as usize];
        let target = nstep_estimator::base(state.gamma, bootstrap, t.clone());
        let first = t.first().unwrap();
        let qvalue = qtable.qvalues(first.obs)[first.action];
        qtable.qvalues_mut(first.obs)[first.action] += state.alpha * (target - qvalue);
    }
}