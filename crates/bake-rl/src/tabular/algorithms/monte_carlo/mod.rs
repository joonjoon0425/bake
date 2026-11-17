//! Monte Carlo

mod on_policy;
mod off_policy;

pub use on_policy::{OnPolicyMonteCarloState, on_policy_monte_carlo_update};
pub use off_policy::OffPolicyMonteCarloState;

use crate::tabular::qtable::QTable;

/// enum for selecting Monte Carlo update (first-visit vs every-visit)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonteCarloVisit {
    /// estimate action value using only the first visit of pair (s, a)
    FirstVisit,
    /// estimate action value using all visit of pair (s, a)
    EveryVisit,
}

/// struct for Monte Carlo statistics for `update` (accumulated return and visit count, etc)
#[derive(Debug, Clone)]
pub struct MonteCarloStat {
    /// visit count of pair (s, a)
    pub visit_count: Vec<usize>,
}

impl MonteCarloStat {
    /// create new `MonteCarloStat`
    pub fn new(qtable: &QTable) -> Self {
        Self { visit_count: vec![0; qtable.n_obs() * qtable.n_action()] }
    }
}