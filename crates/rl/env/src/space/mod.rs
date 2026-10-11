//! A specification of observation and action space
//!
//! 

pub mod space;
pub mod discrete;
pub mod continuous;

pub use space::Space;
pub use discrete::DiscreteSpace;
pub use continuous::ContinuousSpace;

#[cfg(feature = "deep")]
pub mod deep;