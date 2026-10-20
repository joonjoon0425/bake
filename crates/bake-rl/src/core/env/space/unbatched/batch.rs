//! convert the spaces into batched space

use crate::core::{
    Vector, Single, data::Batchable, env::space::{
        ContinuousSpace, DiscreteSpace, Space,
    }
};
use ndarray::{Array, Dim};
use burn::prelude::*;
/// `Space` can be converted to `Space` whose elements are `Batchable`
pub trait ToBatchedSpace: Space<Single> + Sized {
    /// `Batchable` `Space`
    type BatchedSpace: Space<Vector, Element: Batchable>;
    /// create a batch from given unbatched elements
    fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space<Vector>>::Element;
    /// create unbatched elements from batch
    fn unbatch(bundle: <Self::BatchedSpace as Space<Vector>>::Element) -> Vec<Self::Element>;
    /// create a batched space from unbatched space
    fn to_batched_space(space: Self, device: &Device) -> Self::BatchedSpace;
}

impl ToBatchedSpace for DiscreteSpace {
    type BatchedSpace = DiscreteSpace;

    fn batch(bundle: Vec<Self::Element>, device: &Device) -> Tensor<1, Int> {
        Tensor::from_ints(bundle.as_slice(), device)
    }

    fn unbatch(bundle: <Self::BatchedSpace as Space<Vector>>::Element) -> Vec<Self::Element> {
        let vec: Vec<u64> = bundle.try_into_vec_as().unwrap();
        let vec = vec.into_iter().map(|u| u as usize).collect();
        vec
    }

    fn to_batched_space(space: Self, _: &Device) -> Self::BatchedSpace {
        let n = space.n().clone();
        DiscreteSpace::new(n)
    }
}

macro_rules! batch_continuous_space {
    ($d:literal) => {
        impl ToBatchedSpace for ContinuousSpace<$d, Single> {
            type BatchedSpace = ContinuousSpace<$d, Vector>;

            fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space<Vector>>::Element {
                let n = bundle.len();
                let mut shape: Vec<usize> = bundle[0].shape().try_into().unwrap();
                let n_elem: usize = shape.iter().product();
                let mut buf = Vec::with_capacity(n_elem * n);
                for a in &bundle {
                    match a.as_slice() {
                        Some(s) => buf.extend_from_slice(s),
                        None => buf.extend(a.iter().copied()),
                    }
                }
                shape[0] = n;
                Tensor::from_data(TensorData::new(buf, shape), device)
            }

            fn unbatch(bundle: <Self::BatchedSpace as Space<Vector>>::Element) -> Vec<Self::Element> {
                let shape = bundle.shape();
                let v: Vec<f32> = bundle.try_into_vec_as().unwrap();
                let per_env = shape.iter().skip(1).product();
                let mut single: Vec<usize> = shape.iter().copied().collect();
                single[0] = 1;
                let single = single.try_into().unwrap();
                let items: Vec<_> = v
                    .chunks(per_env)
                    .map(|c| Array::from_shape_vec(Dim::<[usize; $d]>(single), c.to_vec()).unwrap())
                    .collect();
                items
            }

            fn to_batched_space(space: Self, device: &Device) -> Self::BatchedSpace {
                let shape: Vec<_> = space.shape().try_into().unwrap();
                let low = space.low().clone();
                let high = space.high().clone();
                let lows = Self::batch(vec![low], device);
                let highs = Self::batch(vec![high], device);
                Self::BatchedSpace::new(&shape, lows, highs)
            }
        }
    };
    ($($d:literal),* $(,)?) => {
        $(
            batch_continuous_space!($d);
        )*
    }
}

batch_continuous_space!(1, 2, 3, 4, 5, 6);