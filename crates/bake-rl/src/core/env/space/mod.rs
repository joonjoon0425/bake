//! Space for specifications of elements
//! - The `Space` corresponds to the `space` of Farama Foundation's `gymnasium`.
use crate::core::data::BatchOps;
use burn::prelude::*;
use rand::Rng;

/// `NativeSpace` trait for specifying the properties of observation space and action space, with `Native` representation
pub trait NativeSpace: Clone {
    /// the type of element
    type Element: Clone + 'static;
    /// sample an random element
    fn sample<R: Rng>(&self, rng: R) -> Self::Element;
}

/// `BurnSpace` trait for specifying the properties of observation space and action space, with `Burn` representation
pub trait BurnSpace: Clone {
    /// the type of element
    type Element: BatchOps;
    /// sample `n` random elements
    fn sample(&self, n: usize) -> Self::Element;
    /// return the device
    fn device(&self) -> burn::prelude::Device;
}

/// conversion from `NativeSpace` to `BurnSpace`
pub trait ToBurnSpace: NativeSpace {
    /// Burn Space
    type BurnSpace: BurnSpace;
    /// convert from `NativeSpace` to `BurnSpace`
    fn into_burn(self, device: &Device) -> Self::BurnSpace;
    /// convert elements to `Burn`
    fn bind(elems: Vec<Self::Element>, device: &Device) -> <Self::BurnSpace as BurnSpace>::Element;
    /// convert elements to `Native`
    fn unbind(elem: <Self::BurnSpace as BurnSpace>::Element) -> Vec<Self::Element>;
}

pub mod discrete;
pub use discrete::DiscreteSpace;

pub mod continuous;
pub use continuous::ContinuousSpace;

pub mod byte;
pub use byte::ByteSpace;
