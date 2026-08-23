//! Encoder trait and basic implementations
//! 

use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};

use crate::types::Batchable;
pub trait Encoder : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

pub mod mlpencoder;
pub use mlpencoder::*;