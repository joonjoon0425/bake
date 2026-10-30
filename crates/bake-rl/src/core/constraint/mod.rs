//! constraint traits and implementations

use crate::core::data::WithBatchDim;

/// basic trait for constraint types: nothing implemented yet
pub trait Constraint: WithBatchDim {}

pub mod discrete;
pub use discrete::DiscreteConstraint;

mod unconstrained;
pub use unconstrained::Unconstrained;