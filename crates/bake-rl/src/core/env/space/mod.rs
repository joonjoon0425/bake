//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
use crate::core::data::BatchOps;
use burn::prelude::*;
use rand::Rng;

/// `Space` trait for specifying the properties of observation space and action space
pub trait Space: Clone {
    /// the type of element
    type Element: Clone + 'static;
    /// sample a random element
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Element;
}

/// `BatchedSpace` trait for specifying the properties of observation space and action space, with `Burn` representation
pub trait BatchedSpace: Clone {
    /// the type of element
    type Element: BatchOps;
    /// sample random elements
    fn sample(&self) -> Self::Element;
    /// return the device
    fn device(&self) -> burn::prelude::Device;
}

/// conversion from `Space` to `BatchedSpace`
pub trait IntoBatchedSpace: Space {
    /// Batched Space
    type Batched: BatchedSpace;
    /// convert from `Space` to `BatchedSpace`
    fn into_batched(spaces: &[Self], device: &Device) -> Self::Batched;
    /// convert elements to batched elements
    fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::Batched as BatchedSpace>::Element;
    /// convert elements to single elements
    fn unbind(elem: <Self::Batched as BatchedSpace>::Element) -> Vec<Self::Element>;
}

pub mod discrete;
pub use discrete::{DiscreteSpace, BatchedDiscreteSpace};

pub mod continuous;
pub use continuous::{ContinuousSpace, BatchedContinuousSpace};

pub mod byte;
pub use byte::{ByteSpace, BatchedByteSpace};
