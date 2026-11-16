//! Monte Carlo
use crate::core::data::Transition;
use crate::core::constraint::DiscreteConstraint;
use crate::tabular::qtable::QTable;

/// state struct for Monte Carlo
pub struct MonteCarloState {
    /// discount rate
    pub gamma: f32,
    /// options for Monte Carlo methods
    pub options: MonteCarloOption,
    
}

/// option struct for Monte Carlo
pub struct MonteCarloOption {
    /// select one of two methods: first-visit Monte Carlo and every-visit Monte Carlo
    pub visit: MonteCarloVisit,
    /// select one of two methods: on-policy MonteCarlo and off-policy MonteCarlo
    pub on_off: MonteCarloOnOff,
}

/// enum for selecting Monte Carlo update (first-visit vs every-visit)
pub enum MonteCarloVisit {
    /// estimate action value using only the first visit of pair (s, a)
    FirstVisit,
    /// estimate action value using all visit of pair (s, a)
    EveryVisit,
}

/// enum for selecting Monte Carlo on-policy and off-policy
pub enum MonteCarloOnOff {
    /// use on-policy method
    OnPolicy,
    /// use off-policy method, using importance sampling
    OffPolicy,
}