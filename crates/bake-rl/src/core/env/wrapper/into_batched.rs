//! converting from non-batch element to batchable elements, which implements `BatchOps` trait
//! # Warning
//! - These are for reward and constraint types, which does not have `Space`

use burn::prelude::*;
use crate::core::data::BatchOps;

pub trait IntoBatched: Sized {
    type Batched: BatchOps;
    /// convert vector of elements to batched representation
    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched;
}

impl IntoBatched for () {
    type Batched = ();

    fn into_batched(_elems: Vec<Self>, _device: &Device) -> Self::Batched {
        ()
    }
}

impl IntoBatched for bool {
    type Batched = Tensor<1, Bool>;

    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_bool(&elems[..], device)
    }
}

impl IntoBatched for f32 {
    type Batched = Tensor<1>;

    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_floats(&elems[..], device)
    }
}

impl IntoBatched for usize {
    type Batched = Tensor<1, Int>;

    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_ints(&elems[..], device)
    }
}

impl<const N: usize> IntoBatched for [bool; N] {
    type Batched = Tensor<2, Bool>;

    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        let n = elems.len();
        let flattened: Vec<bool> = elems.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flattened, [n, N]), device)
    }
}


impl IntoBatched for Vec<bool> {
    type Batched = Tensor<2, Bool>;

    fn into_batched(elems: Vec<Self>, device: &Device) -> Self::Batched {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        // check if all elements has same length
        let n = elems.len();
        let len = elems[0].len();
        assert!(elems.iter().all(|v| v.len() == len), "ToBurn::bind: elements in given Vec<{}> has different lengths", std::any::type_name::<Self>());
        let flattened: Vec<bool> = elems.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flattened, [n, len]), device)
    }
}