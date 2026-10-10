//! A specification of observation and avtion space
//!
 
/// Basic space trait
pub trait Space {
    /// the type of element of the space
    type Element;
}

pub mod discrete;
pub mod continuous;

pub use discrete::DiscreteSpace;
pub use continuous::ContinuousSpace;
