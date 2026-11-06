//! converting from non-batch element to batchable elements, which implements `BatchOps` trait
use burn::prelude::*;
use crate::core::data::BatchOps;

pub trait ToBurn: Sized {
    type Batched: BatchOps;
    /// convert vector of elements to `Burn` representation
    fn bind(elems: Vec<Self>, device: &Device) -> Self::Batched;
}

impl ToBurn for () {
    type Batched = ();

    fn bind(_elems: Vec<Self>, _device: &Device) -> Self::Batched {
        ()
    }
}

impl ToBurn for bool {
    type Batched = Tensor<1, Bool>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_bool(elems.as_slice(), device)
    }
}

impl<const N: usize> ToBurn for [bool; N] {
    type Batched = Tensor<2, Bool>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::Batched {
        
    }
}