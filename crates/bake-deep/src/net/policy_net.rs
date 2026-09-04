//! Network trait for Poliy Gradient methods
//! 
use burn::module::{AutodiffModule, ModuleDisplay};
use crate::data::batchable::Batchable;

/// Policy network trait
pub trait PolicyNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation type
    type Obs: Batchable;
    /// parameters which the policy network produces. Can be used for creating the distributions.
    type Params;

    /// returns the params: logits, (mean, std) whatever
    fn forward(&self, obs: Self::Obs) -> Self::Params;
}