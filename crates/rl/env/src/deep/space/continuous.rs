//! contiuous space

use burn::prelude::*;
use crate::space::{ContinuousSpace, Space};
use crate::deep::space::convert::ToUnbatchedSpace;
/// A tensor space which is continuous
pub struct ContinuousTensorSpace<const D: usize = 2> {
    shape: Vec<usize>,
    low: <Self as Space>::Element,
    high: <Self as Space>::Element,
    device: Device,
}

impl<const D: usize> ContinuousTensorSpace<D> {
    /// create a new tensor continuous space
    pub fn new(shape: Vec<usize>, low: <Self as Space>::Element, high: <Self as Space>::Element) -> Self {
        let device = low.device();
        Self { shape, low, high, device }
    }
    /// return the device
    pub fn device(&self) -> &Device { &self.device }
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

impl<const D: usize> Space for ContinuousTensorSpace<D> {
    type Element = Tensor<D>;
}

macro_rules! unbatched {
    ($db:literal => $d:literal) => {
        impl ToUnbatchedSpace for ContinuousTensorSpace<$db> {
            type UnbatchedSpace = ContinuousSpace<$d>;

            fn unbatched_space(&self) -> Self::UnbatchedSpace {
                debug_assert_eq!(self.low.dims()[0], 1);
                let flattened: Vec<f32> = self.low.clone().into_data().try_into_vec().unwrap();
                let shape = self.shape();
                let low = ndarray::prelude::Array::from_shape_vec(shape, flattened).expect("length mismatch");

                debug_assert_eq!(self.high.dims()[0], 1);
                let flattened: Vec<f32> = self.high.clone().into_data().try_into_vec().unwrap();
                let shape = self.shape();
                let high = ndarray::prelude::Array::from_shape_vec(shape, flattened).expect("length mismatch");

                Self::UnbatchedSpace::new(self.shape().try_into().unwrap(), low, high)
            }

            fn to_batchable(&self, data: <Self::UnbatchedSpace as Space>::Element) -> Self::Element {
                let data: Vec<f32> = data.iter().copied().collect();
                Tensor::<$d>::from_data(TensorData::new(data, self.shape()), &self.device).unsqueeze_dim::<$db>(0)
            }
        }
    };
}

unbatched!(2 => 1);
unbatched!(3 => 2);
unbatched!(4 => 3);
unbatched!(5 => 4);
unbatched!(6 => 5);
unbatched!(7 => 6);
unbatched!(8 => 7);
