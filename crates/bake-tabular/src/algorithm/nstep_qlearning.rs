//! An implementation of tabular n-step Q-learning algorithm
//! 
use crate::{algorithm::nstep_estimator::{self, Bootstrap}, constraint::Constraint, data::Transition, explore::{Exploration, Greedy}, qtable::QTable};

/// A state for QLearning
#[derive(Debug, Clone)]
pub struct NStepQLearning {
    /// n
    pub n: usize,
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
}

impl NStepQLearning {
    /// update the qtable with given state, using n-step Q-learning algorithm with importance weight sampling
    pub fn update_is<C: Constraint>(state: &NStepQLearning, qtable: &mut QTable, exploration: &impl Exploration, t: Vec<Transition<C>>) {
        let target = nstep_estimator::base(qtable, exploration, state.gamma, Bootstrap::Max, t.clone());
        let first = t.first().unwrap();
        let qvalue = qtable.qvalues(first.obs)[first.action];
        // compute the importance sampling weight
        let mut b_log_prob = 0f32;
        let mut pi_log_prob = 0f32;
        // using a new `Greedy` won't matter since the prob does not use the seed or ramdom samplings
        let greedy = Greedy::new(0);
        let mut prob_0 = false;
        for transition in t.iter().skip(1) {
            b_log_prob += exploration.prob(qtable, transition.obs, transition.action, transition.constraint).ln();
            let pi_prob = greedy.prob(qtable, transition.obs, transition.action, transition.constraint);
            if pi_prob == 0f32 { prob_0 = true; break; }
            pi_log_prob += pi_prob.ln();   
        }
        let is_weight = if prob_0 { 0f32 } else { (pi_log_prob - b_log_prob).exp() };
        qtable.qvalues_mut(first.obs)[first.action] += is_weight * state.alpha * (target - qvalue);
    }

    /// update the qtable with given state, using n-step Q-learning algorithm with tree backup method
    pub fn update_tb<C: Constraint>(state: &NStepQLearning, qtable: &mut QTable, t: Vec<Transition<C>>) {
        // using a new `Greedy` won't matter since the prob does not use the seed or ramdom samplings
        let greedy = Greedy::new(0);
        let first = t.first().unwrap().clone();
        let target = nstep_estimator::tree_backup(qtable, &greedy, state.gamma, t);
        let qvalue = qtable.qvalues(first.obs)[first.action];
        qtable.qvalues_mut(first.obs)[first.action] += state.alpha * (target - qvalue);
    }
}