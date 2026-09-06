//! Network trait for Poliy Gradient methods
//! 
use burn::{module::{AutodiffModule, ModuleDisplay}};
use crate::data::batchable::Batchable;

/// Policy network trait
/// - the users must implement this trait to use their own network structure, and wrap it with wrapper.
/// # Warning
/// - the user must create the network initialy on autodiff device.
pub trait PolicyNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// parameters which the policy network produces. Can be used for creating the distributions.
    type Params;

    /// returns the params: logits, (mean, std) whatever
    fn forward(&self, obs: Self::Obs) -> Self::Params;
}