//! traits for spaces which are convertible to Batchable element space
//! # Warning
//! - the spaces are convertible one by one
use bake_rl_core::data::deep::Batchable;
use crate::space::Space;
use burn::prelude::*;

/// Tensor convertible space
pub trait ToBatchedSpace: Space {
    /// the batched space
    type BatchedSpace: Space<Element: Batchable> + ToUnbatchedSpace<UnbatchedSpace = Self>;
    /// return the batched space
    fn batched_space(&self, device: &Device) -> Self::BatchedSpace;
    /// get the batch and turn it into raw type data
    fn to_raw(&self, batch: <Self::BatchedSpace as Space>::Element) -> Self::Element;
}

/// raw convertible space
pub trait ToUnbatchedSpace: Space {
    /// the unbatched space
    type UnbatchedSpace: Space + ToBatchedSpace<BatchedSpace = Self>;
    /// return the unbatched space
    fn unbatched_space(&self) -> Self::UnbatchedSpace;
    /// get the raw data and convert it into batch type data
    fn to_batchable(&self, data: <Self::UnbatchedSpace as Space>::Element) -> Self::Element;
}
