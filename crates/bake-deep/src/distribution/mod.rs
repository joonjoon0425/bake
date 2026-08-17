//! Distribution trait for action selection
use burn::Tensor;
use crate::types::Batchable;
pub trait Distribution {
    type Action: Batchable;
    fn sample(&self) -> Self::Action;
    fn mode(&self) -> Self::Action;

    fn log_probs(&self, action: Self::Action) -> Tensor<1>; // [batch]
    fn entropy(&self) -> Tensor<1>; // [batch]
}

pub mod categorical;
pub use categorical::*;