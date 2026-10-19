//! Space for continuous elements

use crate::core::{Marker, env::space::Space};

/// Continuous space
/// - E: element type
///     - `burn::Tensor`
///     - `ndarray::Array`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<const D: usize, M: Marker> {
    shape: Vec<usize>,
    low: M::FloatArray<D>,
    high: M::FloatArray<D>,
}

impl<const D: usize, M: Marker> ContinuousSpace<D, M> {
    /// create a new continuous space, with `burn::Tensor<D>` element
    pub fn new(shape: &[usize], low: M::FloatArray<D>, high: M::FloatArray<D>) -> Self {
        Self { shape: shape.try_into().unwrap(), low, high }
    }
    /// return the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &M::FloatArray<D> {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &M::FloatArray<D> {
        &self.high
    }
}

impl<const D: usize, M: Marker> Space<M> for ContinuousSpace<D, M> {
    type Element = M::FloatArray<D>;
}