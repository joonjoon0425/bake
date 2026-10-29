//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
//! 
use burn::prelude::*;
use crate::core::data::WithBatchDim;
/// `Space` trait for specifying the properties of observation space and action space
pub trait Space: Clone {
    /// the type of element
    type Element: WithBatchDim;
    /// sample `n` random elements
    fn sample(&self, num: usize) -> Self::Element;
    /// return the device of space
    fn device(&self) -> Device;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub mod byte;
pub use byte::ByteSpace;