//! Policy Network for Poliy Gradient methods
//! 
use burn::module::{AutodiffModule, ModuleDisplay};

use crate::types::Batchable;
pub trait PolicyNet : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    type Params;

    /// returns the params: logits, (mean, std) whatever
    fn forward(&self, obs: Self::Obs) -> Self::Params;
}