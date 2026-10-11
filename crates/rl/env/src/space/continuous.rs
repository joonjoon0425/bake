//! Continuous space trait and its implementations

use crate::space::Space;
use ndarray::prelude::*;

/// NdArray typed continuous space
pub struct ContinuousSpace<const D: usize = 1> {
    shape: [usize; D],
    low: <Self as Space>::Element,
    high: <Self as Space>::Element
}

impl<const D: usize> ContinuousSpace<D> {
    /// create a new continuous ndarray space
    pub fn new(shape: [usize; D], low: <Self as Space>::Element, high: <Self as Space>::Element) -> Self {
        Self { shape, low, high }
    }
    /// return the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &<Self as Space>::Element {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &<Self as Space>::Element {
        &self.high
    }
}

impl<const D: usize> Space for ContinuousSpace<D> {
    type Element = ArrayD<f32>;
}

#[cfg(feature = "deep")]
macro_rules! batched {
    ($d:literal => $db:literal) => {
        impl crate::space::deep::convert::ToBatchedSpace for ContinuousSpace<$d> {
            type BatchedSpace = crate::space::deep::continuous::ContinuousTensorSpace<$db>;
            
            fn batched_space(&self, device: &burn::prelude::Device) -> Self::BatchedSpace {
                let low: Vec<f32> = self.low.iter().copied().collect();
                let low = burn::prelude::Tensor::<$d>::from_data(burn::prelude::TensorData::new(low, self.shape()), device).unsqueeze_dim::<$db>(0);
                let high: Vec<f32> = self.high.iter().copied().collect();
                let high = burn::prelude::Tensor::<$d>::from_data(burn::prelude::TensorData::new(high, self.shape()), device).unsqueeze_dim::<$db>(0);

                Self::BatchedSpace::new(self.shape().to_vec(), low, high)
            }

            fn to_raw(&self, batch: <Self::BatchedSpace as Space>::Element) -> Self::Element {
                debug_assert_eq!(batch.dims()[0], 1);
                let flattened: Vec<f32> = batch.into_data().try_into_vec().unwrap();
                let shape = self.shape();
                Array::from_shape_vec(shape, flattened).expect("length mismatch")
            }
        }
    };
}

#[cfg(feature = "deep")]
batched!(1 => 2);
#[cfg(feature = "deep")]
batched!(2 => 3);
#[cfg(feature = "deep")]
batched!(3 => 4);
#[cfg(feature = "deep")]
batched!(4 => 5);
#[cfg(feature = "deep")]
batched!(5 => 6);
#[cfg(feature = "deep")]
batched!(6 => 7);
#[cfg(feature = "deep")]
batched!(7 => 8);