//! computes the n-step results for n-step methods
//! 

use crate::{constraint::Constraint, data::Transition, explore::Exploration, qtable::{QTable, QValues}};

/// computes the n-step estimate of returns
#[derive(Debug, Clone, Copy)]
pub enum NStepEstimator {
    /// A base estimator with no importance sampling weight, tree backup. Basically for on-policy methods
    Base,
    /// The n-step estimator with tree-backup method
    TreeBackup,
}

/// decides which bootstrap method to use
#[derive(Debug, Clone, Copy)]
pub enum Bootstrap {
    /// bootstrap with maximum q value
    Max,
    /// boostrap with next action
    NextAction,
    /// bootstrap with expectation
    Expectation,
}

impl Bootstrap {
    /// return the bootstrap value
    pub fn bootstrap<C: Constraint>(&self, qtable: &QTable, exploration: &impl Exploration, t: Transition<C>) -> f32 {
        match self {
            Bootstrap::Max => { qtable.qvalues(t.next_obs).max(t.next_constraint) },
            Bootstrap::NextAction => { qtable.qvalues(t.next_obs)[t.get("next_action").unwrap() as usize] },
            Bootstrap::Expectation => { qtable.expectation(exploration, t.next_obs, t.next_constraint) }
        }
    }
}


impl NStepEstimator {
    /// return the n-step estimate of return
    pub fn estimate<C: Constraint, E: Exploration>(&self, qtable: &QTable, exploration: &E, gamma: f32, bootstrap: Bootstrap, t: Vec<Transition<C>>) -> f32 {
        match self {
            NStepEstimator::Base => self.base(qtable, exploration, gamma, bootstrap, t),
            NStepEstimator::TreeBackup => self.tree_backup(qtable, exploration, gamma, t)
        }
    }

    fn base<C: Constraint, E: Exploration>(&self, qtable: &QTable, exploration: &E, gamma: f32, bootstrap: Bootstrap, t: Vec<Transition<C>>) -> f32 {
        let last = t.last().unwrap();
        let bootstrap = bootstrap.bootstrap(qtable, exploration, last.clone());
        
        let mut target = if last.terminated { 0f32 } else { bootstrap };

        for transition in t.iter().rev() {
            target = gamma * target + transition.reward;
        }
        target
    }

    fn tree_backup<C: Constraint, E: Exploration>(&self, qtable: &QTable, exploration: &E, gamma: f32, t: Vec<Transition<C>>) -> f32 {
        let last = t.last().unwrap();
        
        let mut target = last.reward + if last.terminated { 0f32 } else {
            gamma * qtable.expectation(exploration, last.next_obs, last.next_constraint)
        };

        for (i, transition) in t.iter().enumerate().skip(1).rev() {
            let obs = transition.obs;
            let constraint = transition.constraint;
            let action = transition.action;
            let qvalues = qtable.qvalues(obs);
            target = t[i - 1].reward + gamma * (qtable.expectation(exploration, obs, constraint) + exploration.prob(qtable, obs, action, constraint) * (target - qvalues[action]))
        }
        target
    }
}