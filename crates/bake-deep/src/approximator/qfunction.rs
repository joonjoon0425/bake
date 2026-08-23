//! A QNetwork trait for value-based methods, and basic helper for creating a new QNetwork
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::{constraint::DiscreteConstraint, types::Batchable};

/// A QNetwork trait for value-based methods
pub trait QFunction : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2>;
}