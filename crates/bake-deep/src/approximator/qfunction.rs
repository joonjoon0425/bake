//! A QFunction trait for value-based methods
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::{constraint::DiscreteConstraint, types::Batchable};

/// A QFunction trait for value-based methods
pub trait QFunction : AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;

    /// get the q values of given observation with current approximator
    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2>;
}