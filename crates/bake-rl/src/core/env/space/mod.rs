//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
//! 

/// `Space` trait for specifying the properties of element
pub trait Space {
    /// Element type
    type Element: Clone + 'static;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub(crate) mod switch;

pub mod unbatched;