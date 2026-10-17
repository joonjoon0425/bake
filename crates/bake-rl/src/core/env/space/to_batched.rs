//! A trait for switching unbatched element to batched element

use crate::core::data::Batchable;
use burn::prelude::*;
use ndarray::{Array, Dim};
/// The type can be turned into `Batchable`
pub trait ToBatched: Clone + Send + Sync + Sized + 'static {
    /// the `Batchable` type of which the `Vec<Self>` is converted to
    type Batched: Batchable;// + ToUnBatched<UnBatched = Self>;
    /// convert to `Self::Batched`
    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched;
}

impl ToBatched for f32 {
    type Batched = Tensor<1>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_floats(bundle.as_slice(), device)
    }
}

impl ToBatched for bool {
    type Batched = Tensor<1, Bool>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        Tensor::from_bool(bundle.as_slice(), device)
    }
}

impl ToBatched for Vec<bool> {
    type Batched = Tensor<2, Bool>;

    fn batch(bundle: Vec<Self>, device: &Device) -> Self::Batched {
        let len = bundle[0].len();
        let n = bundle.len();
        Tensor::from_data(TensorData::new(bundle.into_iter().flatten().collect(), [n, len]), device)
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
batch_ndarray!(1, 2, 3, 4, 5, 6);
// macro_rules! unbatch_ndarray {
//     ($d:literal) => {
//         impl ToUnBatched for Tensor<{$d + 1}> {
//             type UnBatched = Array<f32, Dim<[usize; $d]>>;

//             fn unbatch(self) -> Vec<Self::UnBatched> {
//                 let shape = self.shape();
//                 let v: Vec<f32> = self.try_into_vec_as().unwrap();
//                 let per_env = shape.iter().skip(1).product();
//                 let single: Vec<usize> = shape.iter().skip(1).copied().collect();
//                 let single = single.try_into().unwrap();
//                 let items: Vec<_> = v
//                     .chunks(per_env)
//                     .map(|c| Array::from_shape_vec(Dim::<[usize; $d]>(single), c.to_vec()).unwrap())
//                     .collect();
//                 items
//             }
//         }
//     };
//     ($($d:literal),* $(,)?) => {
//         $(
//             unbatch_ndarray!($d);
//         )*
//     }
// }


// unbatch_ndarray!(1, 2, 3, 4, 5, 6);