//! Network trait for value based methods
//! 
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::data::batchable::Batchable;

/// A discrete QNetwork trait
pub trait DiscreteQNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// returns the raw q values
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

/// A discrete Dueling QNetwork trait
pub trait DiscreteDuelingQNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// returns the raw value and advantage
    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>);
}