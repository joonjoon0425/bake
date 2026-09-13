//! Network trait for value based methods
//! 
use burn::{prelude::*, module::{AutodiffModule, ModuleDisplay}};
use crate::data::batchable::Batchable;

/// A discrete QNetwork trait
/// - the users must implement this trait to use their own network structure, and wrap it with wrapper.
/// # Warning
/// - the user must create the network initialy on autodiff device.
pub trait DiscreteQNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// returns the raw q values
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

/// A discrete Dueling QNetwork trait
/// - the users must implement this trait to use their own network structure, and wrap it with wrapper.
/// # Warning
/// - the user must create the network initialy on autodiff device.
pub trait DiscreteDuelingQNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// returns the raw value and advantage
    fn forward(&self, obs: Self::Obs) -> (Tensor<1>, Tensor<2>);
}