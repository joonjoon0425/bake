//! A trait for switching between batched and unbatched elements

use crate::core::data::Batchable;
use burn::prelude::*;
use ndarray::{Array, Dim};
/// The type can be turned into `Batchable`
pub trait ToBatched: Sized + 'static {
    /// the `Batchable` type of which the `Vec<Self>` is converted to
    type Batched: Batchable;// + ToUnBatched<UnBatched = Self>;
    /// convert to `Self::Batched`
    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched;
}

/// the type that can be unbatched
pub trait ToUnBatched: Batchable {
    /// the individual unbatched type
    type UnBatched;// : ToBatched<Batched = Self>;
    /// convert to `Self::UnBatched`
    fn unbatch(self) -> Vec<Self::UnBatched>;
}
// /// blanket impl
// impl<T: Batchable> ToBatchable for T {
//     type Batchable = T;

//     /// note that `Batchable` type is already `Batchable`
//     /// # Warning
//     /// - so the given `device` is not used. That is, it is not transferred to given `device`
//     fn to_batch(bundle: Vec<Self>, _: &Device) -> Self::Batchable {
//         T::cat(bundle)
//     }
// }

impl ToBatched for usize {
    type Batched = Tensor<1, Int>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_ints(bundle.as_slice(), device)
    }
}

impl ToUnBatched for Tensor<1, Int> {
    type UnBatched = usize;

    fn unbatch(self) -> Vec<Self::UnBatched> {
        let vec: Vec<u64> = self.try_into_vec_as().unwrap();
        let vec = vec.into_iter().map(|u| u as usize).collect();
        vec
    }
}

impl ToBatched for f32 {
    type Batched = Tensor<1>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_floats(bundle.as_slice(), device)
    }
}

impl ToUnBatched for Tensor<1> {
    type UnBatched = f32;

    fn unbatch(self) -> Vec<Self::UnBatched> {
        self.try_into_vec_as().unwrap()
    }
}

impl ToBatched for bool {
    type Batched = Tensor<1, Bool>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_bool(bundle.as_slice(), device)
    }
}

impl ToUnBatched for Tensor<1, Bool> {
    type UnBatched = bool;

    fn unbatch(self) -> Vec<Self::UnBatched> {
        self.try_into_vec_as().unwrap()
    }
}

impl ToBatched for Vec<bool> {
    type Batched = Tensor<2, Bool>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        let len = bundle[0].len();
        Tensor::from_data(TensorData::new(bundle.into_iter().flatten().collect(), [1, len]), device)
    }
}

impl ToUnBatched for Tensor<2, Bool> {
    type UnBatched = Vec<bool>;

    fn unbatch(self) -> Vec<Self::UnBatched> {
        let shape = self.shape();
        let (n, action_num) = (shape[0], shape[1]);
        let v: Vec<bool> = self.try_into_vec_as::<bool>().unwrap();
        debug_assert_eq!(v.len(), n * action_num);
        v.chunks_exact(action_num).map(|row| row.to_vec()).collect()
    }
}

macro_rules! batch_ndarray {
    ($d:literal) => {
        impl ToBatched for Array<f32, Dim<[usize; $d]>> {
            type Batched = Tensor<{$d + 1}>;

            fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
                let n = bundle.len();
                let shape = bundle[0].shape();
                let n_elem: usize = shape.iter().product();
                let mut buf = Vec::with_capacity(n_elem * n);
                for a in &bundle {
                    match a.as_slice() {
                        Some(s) => buf.extend_from_slice(s),
                        None => buf.extend(a.iter().copied()),
                    }
                }
                let final_shape: Vec<_> = [n].iter().chain(shape.iter()).map(|f| *f).collect();
                Tensor::from_data(TensorData::new(buf, final_shape), device)
            }
        }
    };
    ($($d:literal),* $(,)?) => {
        $(
            batch_ndarray!($d);
        )*
    }
}

macro_rules! unbatch_ndarray {
    ($d:literal) => {
        impl ToUnBatched for Tensor<{$d + 1}> {
            type UnBatched = Array<f32, Dim<[usize; $d]>>;

            fn unbatch(self) -> Vec<Self::UnBatched> {
                let shape = self.shape();
                let v: Vec<f32> = self.try_into_vec_as().unwrap();
                let per_env = shape.iter().skip(1).product();
                let single: Vec<usize> = shape.iter().skip(1).copied().collect();
                let single = single.try_into().unwrap();
                let items: Vec<_> = v
                    .chunks(per_env)
                    .map(|c| Array::from_shape_vec(Dim::<[usize; $d]>(single), c.to_vec()).unwrap())
                    .collect();
                items
            }
        }
    };
    ($($d:literal),* $(,)?) => {
        $(
            unbatch_ndarray!($d);
        )*
    }
}

batch_ndarray!(1, 2, 3, 4, 5, 6);
unbatch_ndarray!(1, 2, 3, 4, 5, 6);