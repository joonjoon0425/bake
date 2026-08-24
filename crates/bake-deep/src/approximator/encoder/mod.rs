//! Encoder trait and basic implementations
//! 

use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::types::Batchable;
/// Encoder trait for helper modules
pub trait Encoder : AutodiffModule + Clone + ModuleDisplay {
    /// the observation which encoder recieves
    type Obs: Batchable;
    /// get the encoded observation
    fn forward(&self, obs: Self::Obs) -> Tensor<2>;
}

pub mod mlpencoder;
pub use mlpencoder::*;