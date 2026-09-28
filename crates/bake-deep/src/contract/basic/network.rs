//! A common trait for user-implemented networks
//! 

use burn::module::{Module, ModuleDisplay};
use crate::data::Batchable;

/// A common trait for user-implemented networks
pub trait Network : Module + ModuleDisplay + Clone {
    /// the observation which the network takes
    type Obs: Batchable;
}