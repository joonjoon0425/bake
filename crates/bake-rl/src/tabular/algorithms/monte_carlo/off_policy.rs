use crate::core::constraint::DiscreteConstraint;
use crate::core::data::Transition;
use crate::tabular::qtable::QTable;
use super::MonteCarloVisit;

/// state struct for off-policy Monte Carlo
#[derive(Debug, Clone)]
pub struct OffPolicyMonteCarloState {
    /// discount rate
    pub gamma: f32,
    /// select one of two methods: first-visit Monte Carlo and every-visit Monte Carlo
    pub visit: MonteCarloVisit,
    /// saves the accumulated returns of episode starting from pair (s, a)
    pub return_accumulated: Vec<f32>,
    /// saves the number of visit on pair (s, a)
    pub visit_count: Vec<usize>
}

/// update q table using off-policy Monte Carlo algorithm
pub fn off_policy_monte_carlo_update(state: &mut OffPolicyMonteCarloState, qtable: &mut QTable, episode: Vec<Transition<usize, impl DiscreteConstraint, usize, f32>>) {

}