//! Space for continuous elements

use crate::core::{Repr, env::space::Space};
use burn::prelude::*;
/// Continuous space
/// - E: element type
///     - `burn::Tensor`
///     - `ndarray::Array`
#[derive(Debug, Clone)]
pub struct ContinuousSpace<const D: usize, M: Repr> {
    shape: Vec<usize>,
    low: M::Array<D, Float>,
    high: M::Array<D, Float>,
}

impl<const D: usize, M: Repr> ContinuousSpace<D, M> {
    /// create a new continuous space, with `burn::Tensor<D>` element
    pub fn new(shape: &[usize], low: M::Array<D, Float>, high: M::Array<D, Float>) -> Self {
        Self { shape: shape.try_into().unwrap(), low, high }
    }
    /// return the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// return the lower bound
    pub fn low(&self) -> &M::Array<D, Float> {
        &self.low
    }
    /// return the upper bound
    pub fn high(&self) -> &M::Array<D, Float> {
        &self.high
    }
}

impl<const D: usize, M: Repr> Space<M> for ContinuousSpace<D, M> {
    type Element = M::Array<D, Float>;
}