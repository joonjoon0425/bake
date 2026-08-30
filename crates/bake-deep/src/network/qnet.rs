//! A Q-Network trait
//! 
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::types::Batchable;

/// A QNetwork trait
pub trait QNet : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    /// returns the raw q values
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

/// A Dueling QNetwork trait
pub trait DuelingQNet : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    /// returns the raw value and advantage
    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>);
}