//! Monte Carlo

mod on_policy;
mod off_policy;

pub use on_policy::{OnPolicyMonteCarloState, OnPolicyMonteCarloStat, on_policy_monte_carlo_update};
pub use off_policy::{OffPolicyMonteCarloState, OffPolicyMonteCarloStat, off_policy_monte_carlo_update};

/// enum for selecting Monte Carlo update (first-visit vs every-visit)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonteCarloVisit {
    /// estimate action value using only the first visit of pair (s, a)
    FirstVisit,
    /// estimate action value using all visit of pair (s, a)
    EveryVisit,
}