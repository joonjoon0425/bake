//! A Batchable trait
use burn::{Tensor, tensor::{Bool, Device, Int, TensorData}};

use crate::types::DiscreteMask;

/// Can translate themselves into batch type
pub trait Batchable : Sized + Clone {
    type Batched : Sized + Clone;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched;
}

impl Batchable for f32 {
    type Batched = Tensor<1>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_floats(data.as_slice(), device)
    }
}

impl Batchable for i64 {
    type Batched = Tensor<1, Int>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_ints(data.as_slice(), device)
    }
}

impl Batchable for bool {
    type Batched = Tensor<1, Bool>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_bool(data.as_slice(), device)
    }
}

impl<const D: usize> Batchable for [f32; D] {
    type Batched = Tensor<2>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        let n = data.len();
        let flat: Vec<f32> = data.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for [i64; D] {
    type Batched = Tensor<2, Int>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        let n = data.len();
        let flat: Vec<i64> = data.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

impl<const D: usize> Batchable for DiscreteMask<D> {
    type Batched = Tensor<2, Bool>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        let n = data.len();
        let flat: Vec<bool> = data.into_iter().map(|m| m.0 ).flatten().collect();
        Tensor::from_data(TensorData::new(flat, [n, D]), device)
    }
}

// The Tensor batchables assume that the tensor already has batch dimension
impl<const D: usize> Batchable for Tensor<D> {
    type Batched = Self;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::cat(data, 0).to_device(device)
    }
}

impl<const D: usize> Batchable for Tensor<D, Int> {
    type Batched = Self;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::cat(data, 0).to_device(device)
    }
}

impl<const D: usize> Batchable for Tensor<D, Bool> {
    type Batched = Self;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::cat(data, 0).to_device(device)
    }
}

impl Batchable for () {
    type Batched = ();

    fn batch(_bundle: Vec<Self>, _device: &Device) -> Self::Batched {}
}