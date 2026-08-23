//! Head trait and basic implementations
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::types::Batchable;

pub trait Head : AutodiffModule + Clone + ModuleDisplay {
    type Output;
    type Constraint: Batchable;
    fn forward(&self, encoded: Tensor<2>, constraint: Self::Constraint) -> Self::Output;
}

pub mod qhead;
pub use qhead::*;

pub mod categoricalhead;
pub use categoricalhead::*;

pub mod vhead;
pub use vhead::*;

