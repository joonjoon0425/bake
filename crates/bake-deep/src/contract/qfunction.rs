//! QFunction traits for value-based methods
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::constraint::discrete_constraint::*;
use crate::data::batchable::Batchable;


/// A QFunction trait for value-based methods, with discrete actions
pub trait DiscreteQFunction : AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;

    /// get the q values of given observation with current approximator
    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2>;
}