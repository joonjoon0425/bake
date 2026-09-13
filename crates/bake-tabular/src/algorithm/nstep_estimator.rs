//! computes the n-step results for n-step methods
//! 

use crate::{constraint::Constraint, data::Transition, explore::Exploration, qtable::QTable};

/// compute the n-step return estimator
pub fn base<C: Constraint>(gamma: f32, bootstrap: f32, t: Vec<Transition<C>>) -> f32 {
    let last = t.last().unwrap();
    let mut target = if last.terminated { 0f32 } else { bootstrap };

    for transition in t.iter().rev() {
        target = gamma * target + transition.reward;
    }
    target
}

/// compute the n-step estimator with tree-backup method
pub fn tree_backup<C: Constraint, E: Exploration>(qtable: &QTable, target_policy: &E, gamma: f32, t: Vec<Transition<C>>) -> f32 {
    let last = t.last().unwrap();
    
    let mut target = last.reward + if last.terminated { 0f32 } else {
        gamma * qtable.expectation(target_policy, last.next_obs, last.next_constraint)
    };

    for (i, transition) in t.iter().enumerate().skip(1).rev() {
        let obs = transition.obs;
        let constraint = transition.constraint;
        let action = transition.action;
        let qvalues = qtable.qvalues(obs);
        target = t[i - 1].reward + gamma * (qtable.expectation(target_policy, obs, constraint) + target_policy.prob(qtable, obs, action, constraint) * (target - qvalues[action]))
    }
    target
}