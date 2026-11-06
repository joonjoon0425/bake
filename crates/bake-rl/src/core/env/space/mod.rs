//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.

/// `Space` trait for specifying the properties of observation space and action space
pub trait Space: Clone {
    /// the type of element
    type Element: 'static;
    /// sample a random element
    fn sample(&mut self) -> Self::Element;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub mod byte;
pub use byte::ByteSpace;