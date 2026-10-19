//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
//! 

use crate::core::Marker;

/// `Space` trait for specifying the properties of observation space and action space
pub trait Space<M: Marker>{
    type Element: Clone + 'static;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub mod to_batched;

pub mod unbatched;