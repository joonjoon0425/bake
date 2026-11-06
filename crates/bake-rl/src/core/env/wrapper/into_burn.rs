//! converting from non-batch element to batchable elements, which implements `BatchOps` trait
//! # Warning
//! - These are for reward and constraint types, which does not have `Space`

use burn::prelude::*;
use crate::core::data::BatchOps;

pub trait ToBurn: Sized {
    type BurnElem: BatchOps;
    /// convert vector of elements to `Burn` representation
    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem;
}

impl ToBurn for () {
    type BurnElem = ();

    fn bind(_elems: Vec<Self>, _device: &Device) -> Self::BurnElem {
        ()
    }
}

impl ToBurn for bool {
    type BurnElem = Tensor<1, Bool>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_bool(&elems[..], device)
    }
}

impl ToBurn for f32 {
    type BurnElem = Tensor<1>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_floats(&elems[..], device)
    }
}

impl ToBurn for usize {
    type BurnElem = Tensor<1, Int>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        Tensor::from_ints(&elems[..], device)
    }
}

impl<const N: usize> ToBurn for [bool; N] {
    type BurnElem = Tensor<2, Bool>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        let n = elems.len();
        let flattened: Vec<bool> = elems.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flattened, [n, N]), device)
    }
}


impl ToBurn for Vec<bool> {
    type BurnElem = Tensor<2, Bool>;

    fn bind(elems: Vec<Self>, device: &Device) -> Self::BurnElem {
        assert!(!elems.is_empty(), "ToBurn::bind: given Vec<{}> is empty", std::any::type_name::<Self>());
        // check if all elements has same length
        let n = elems.len();
        let len = elems[0].len();
        assert!(elems.iter().all(|v| v.len() == len), "ToBurn::bind: elements in given Vec<{}> has different lengths", std::any::type_name::<Self>());
        let flattened: Vec<bool> = elems.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flattened, [n, len]), device)
    }
}