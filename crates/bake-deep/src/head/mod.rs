//! Head trait and basic implementations
use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};

pub trait Head : AutodiffModule + Clone + ModuleDisplay {
    type Output;
    type Constraint: Batchable;
    fn forward(&self, encoded: Tensor<2>, constraint: Self::Constraint) -> Self::Output;
}

pub trait QHead: AutodiffModule + Clone + ModuleDisplay {
    fn forward(&self, encoded: Tensor<2>, constraint: impl DiscreteConstraint) -> Tensor<2>;
}

pub trait VHead: AutodiffModule + Clone + ModuleDisplay {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1>;
}

pub mod qhead;
pub use qhead::*;

pub mod categoricalhead;
pub use categoricalhead::*;

pub mod valuehead;
pub use valuehead::*;

use crate::types::{Batchable, DiscreteConstraint};