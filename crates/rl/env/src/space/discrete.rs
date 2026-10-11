//! Discrete Space trait and its implementations

use crate::space::Space;

/// usize typed Discrete Space
pub struct DiscreteSpace { n: usize }
impl DiscreteSpace {
    /// create a new discrete space
    pub fn new(n: usize) -> Self { Self { n } }
    /// return the number of elements
    pub fn n(&self) -> usize {
        self.n
    }
}

impl Space for DiscreteSpace { type Element = usize; }

#[cfg(feature = "deep")]
impl crate::space::deep::convert::ToBatchedSpace for DiscreteSpace {
    type BatchedSpace = crate::space::deep::discrete::DiscreteTensorSpace;

    fn batched_space(&self, device: &burn::prelude::Device) -> Self::BatchedSpace {
        crate::space::deep::discrete::DiscreteTensorSpace::new(self.n(), device)
    }

    fn to_raw(&self, batch: <Self::BatchedSpace as Space>::Element) -> Self::Element {
        batch.into_scalar::<u64>() as usize
    }
}