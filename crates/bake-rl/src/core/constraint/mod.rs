//! constraint traits and implementations

pub mod discrete;
pub use discrete::DiscreteConstraint;

mod unconstrained;
pub use unconstrained::Unconstrained;