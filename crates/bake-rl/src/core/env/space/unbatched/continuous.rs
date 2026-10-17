//! single continuous space

use ndarray::{Array, Dim};

use crate::core::env::space::Space;

/// single continuous space
pub struct SingleContinuousSpace<const D: usize> {
    shape: Vec<usize>,
    low: <Self as Space>::Element,
    high: <Self as Space>::Element
}

impl<const D: usize> SingleContinuousSpace<D> {
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

impl<const D: usize> Space for SingleContinuousSpace<D> {
    type Element = Array<f32, Dim<[usize; D]>>;
}