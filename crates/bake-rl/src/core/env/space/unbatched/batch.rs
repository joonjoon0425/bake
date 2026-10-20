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
    /// create a batched space from unbatched spaces
    fn to_batched_space(spaces: Vec<Self>, device: &Device) -> Self::BatchedSpace;
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

    fn to_batched_space(spaces: Vec<Self>, _: &Device) -> Self::BatchedSpace {
        DiscreteSpace::new(spaces[0].n())
    }
}

macro_rules! batch_continuous_space {
    ($d:literal) => {
        impl ToBatchedSpace for ContinuousSpace<$d, Single> {
            type BatchedSpace = ContinuousSpace<$d, Vector>;

            fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space<Vector>>::Element {
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

            fn unbatch(bundle: <Self::BatchedSpace as Space<Vector>>::Element) -> Vec<Self::Element> {
                let shape = bundle.shape();
                let v: Vec<f32> = bundle.try_into_vec_as().unwrap();
                let per_env = shape.iter().skip(1).product();
                let single: Vec<usize> = shape.iter().skip(1).copied().collect();
                let single = single.try_into().unwrap();
                let items: Vec<_> = v
                    .chunks(per_env)
                    .map(|c| Array::from_shape_vec(Dim::<[usize; $d]>(single), c.to_vec()).unwrap())
                    .collect();
                items
            }

            fn to_batched_space(spaces: Vec<Self>, device: &Device) -> Self::BatchedSpace {
                let mut shape: Vec<_> = spaces[0].shape().try_into().unwrap();
                let n = spaces.len();
                shape[0] = n;
                let lows: Vec<<Self as Space<Single>>::Element> = spaces.iter().map(|space| space.low().clone()).collect();
                let highs: Vec<<Self as Space<Single>>::Element> = spaces.iter().map(|space| space.high().clone()).collect();
                let lows = Self::batch(lows, device);
                let highs = Self::batch(highs, device);
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