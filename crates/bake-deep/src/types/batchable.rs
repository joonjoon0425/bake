//! A Batchable trait
use burn::{Tensor, tensor::{Bool, Device, Int, TensorData}};

use crate::types::{BatchedDiscreteMask, BatchedNoMask, DiscreteMask, NoMask};

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

impl<const ACTION_NUM: usize> Batchable for DiscreteMask<ACTION_NUM> {
    type Batched = BatchedDiscreteMask<ACTION_NUM>;

    fn batch(data: Vec<Self>, device: &Device) -> Self::Batched {
        let data: Vec<Tensor<1, Bool>> = data.into_iter().map(|t| {t.0}).collect();
        BatchedDiscreteMask(Tensor::stack(data, 0).to_device(device))
    }
}

impl Batchable for NoMask {
    type Batched = BatchedNoMask;

    fn batch(_: Vec<Self>, _: &Device) -> Self::Batched { BatchedNoMask }
}

macro_rules! impl_batchable_tensor {
    ($d:literal => $db:literal) => {
        impl Batchable for Tensor<$d> {
            type Batched = Tensor<$db>;

            fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
                Tensor::stack(bundle, 0).to_device(device)
            }
        }

        impl Batchable for Tensor<$d, Int> {
            type Batched = Tensor<$db, Int>;

            fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
                Tensor::stack(bundle, 0).to_device(device)
            }
        }

        impl Batchable for Tensor<$d, Bool> {
            type Batched = Tensor<$db, Bool>;

            fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
                Tensor::stack(bundle, 0).to_device(device)
            }
        }
    };
}

impl_batchable_tensor!(1 => 2);
impl_batchable_tensor!(2 => 3);
impl_batchable_tensor!(3 => 4);
impl_batchable_tensor!(4 => 5);
impl_batchable_tensor!(5 => 6);
impl_batchable_tensor!(6 => 7);
impl_batchable_tensor!(7 => 8);
impl_batchable_tensor!(8 => 9);

impl Batchable for () {
    type Batched = ();

    fn batch(_bundle: Vec<Self>, _device: &Device) -> Self::Batched {}
}