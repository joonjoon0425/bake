//! discrete space
//! 
use burn::prelude::*;

use crate::space::{DiscreteSpace, Space};
use crate::deep::space::convert::ToUnbatchedSpace;

/// A tensor space which is discrete
pub struct DiscreteTensorSpace {
    n: usize,
    device: Device,
}

impl DiscreteTensorSpace {
    /// create a new tensor discrete space
    pub fn new(n: usize, device: &Device) -> Self { Self { n, device: device.clone() } }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
    /// return the device
    pub fn device(&self) -> &Device { &self.device }
}

impl Space for DiscreteTensorSpace {
    type Element = Tensor<1, Int>;
}

impl ToUnbatchedSpace for DiscreteTensorSpace {
    type UnbatchedSpace = DiscreteSpace;

    fn unbatched_space(&self) -> Self::UnbatchedSpace {
        DiscreteSpace::new(self.n())
    }

    fn to_batchable(&self, data: <Self::UnbatchedSpace as Space>::Element) -> Self::Element {
        Tensor::<1, Int>::from_ints([data], &self.device())
    }
}