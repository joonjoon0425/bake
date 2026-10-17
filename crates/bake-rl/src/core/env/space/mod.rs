//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
//! 

/// `Space` trait for specifying the properties of element
pub trait Space: Clone {
    /// Element type
    type Element: Clone + 'static;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub mod to_batched;

pub mod unbatched;