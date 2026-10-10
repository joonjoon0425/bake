//! Continuous space trait and its implementations

use crate::space::Space;
use ndarray::prelude::*;

/// Continuous Space
pub trait ContinuousSpace: Space {
    /// returns the shape of element
    fn shape(&self) -> &Vec<usize>;
    /// returns the lower bound of element
    fn low(&self) -> &Self::Element;
    /// returns the upper bound of element
    fn high(&self) -> &Self::Element;
}

/// NdArray typed continuous space
pub struct NdArrayContinuousSpace {
    shape: Vec<usize>,
    low: <Self as Space>::Element,
    high: <Self as Space>::Element
}

impl NdArrayContinuousSpace {
    pub fn new(shape: Vec<usize>, low: <Self as Space>::Element, high: <Self as Space>::Element) -> Self {
        Self { shape, low, high }
    }
}

impl Space for NdArrayContinuousSpace {
    type Element = ArrayD<f32>;
}

impl ContinuousSpace for NdArrayContinuousSpace {
    fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    fn low(&self) -> &Self::Element {
        &self.low
    }

    fn high(&self) -> &Self::Element {
        &self.high
    }
}