//! Head trait and basic implementations
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::types::ActionMask;

pub trait Head : AutodiffModule + Clone + ModuleDisplay {
    type Output;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, encoded: Tensor<2>, mask: M, fill_value: f32) -> Self::Output;
}

pub mod qhead;
pub use qhead::*;

pub mod categoricalhead;
pub use categoricalhead::*;