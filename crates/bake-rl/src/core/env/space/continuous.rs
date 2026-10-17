//! Space for continuous elements

use crate::core::env::space::Space;
use burn::prelude::*;

/// Continuous space
pub struct ContinuousSpace<const D: usize> {
    shape: Vec<usize>,
    low: <Self as Space>::Element,
    high: <Self as Space>::Element
}

impl<const D: usize> ContinuousSpace<D> {
    /// create a new continuous space
    pub fn new(shape: &[usize], low: <Self as Space>::Element, high: <Self as Space>::Element) -> Self {
        Self { shape: shape.try_into().unwrap(), low, high }
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
    type Element = Tensor<D>;
}