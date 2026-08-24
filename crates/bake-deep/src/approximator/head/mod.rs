//! Head trait and basic implementations
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::types::Batchable;

/// basic trait for heads
pub trait Head : AutodiffModule + Clone + ModuleDisplay {
    /// the result which head peoduces
    type Output;
    /// the constraint
    type Constraint: Batchable;
    /// produce the result of head
    fn forward(&self, encoded: Tensor<2>, constraint: Self::Constraint) -> Self::Output;
}

pub mod qhead;
pub use qhead::*;

pub mod categoricalhead;
pub use categoricalhead::*;

pub mod vhead;
pub use vhead::*;

