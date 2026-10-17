//! convert the spaces into batched space

use crate::core::{
    data::Batchable,
    env::space::{
        ContinuousSpace, DiscreteSpace, Space,
        switch::{ToBatched, ToUnBatched},
        unbatched::{continuous::SingleContinuousSpace, discrete::SingleDiscreteSpace}
    }
};
use burn::prelude::*;
pub trait ToBatchedSpace: Space {
    type BatchedSpace: Space<Element: Batchable>;

    fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space>::Element;
    fn unbatch(bundle: <Self::BatchedSpace as Space>::Element) -> Vec<Self::Element>;
}

impl ToBatchedSpace for SingleDiscreteSpace {
    type BatchedSpace = DiscreteSpace;

    fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space>::Element {
        ToBatched::batch(bundle, device)
    }

    fn unbatch(bundle: <Self::BatchedSpace as Space>::Element) -> Vec<Self::Element> {
        ToUnBatched::unbatch(bundle)
    }
}

macro_rules! batch_continuous_space {
    ($d:literal) => {
        impl ToBatchedSpace for SingleContinuousSpace<$d> {
            type BatchedSpace = ContinuousSpace<{$d + 1}>;

            fn batch(bundle: Vec<Self::Element>, device: &Device) -> <Self::BatchedSpace as Space>::Element {
                ToBatched::batch(bundle, device)
            }

            fn unbatch(bundle: <Self::BatchedSpace as Space>::Element) -> Vec<Self::Element> {
                ToUnBatched::unbatch(bundle)
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