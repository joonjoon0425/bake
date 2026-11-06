//! Space for discrete elements

use burn::prelude::*;
use rand::prelude::*;

use super::{NativeSpace, BurnSpace};

/// Discrete space
#[derive(Debug, Clone)]
pub struct DiscreteSpace<Repr: DiscreteSpaceRepr> {
    n: usize,
    inner: Repr::Inner,
}

/// Discrete Space Representation
pub trait DiscreteSpaceRepr {
    /// inner types which each representation requires
    type Inner;
}

impl DiscreteSpaceRepr for crate::core::Native { type Inner = SmallRng; }
impl DiscreteSpaceRepr for crate::core::Burn { type Inner = Device; }

impl DiscreteSpace<crate::core::Burn> {
    /// create a new `DiscreteSpace` with `Burn` representation
    pub fn burn(n: usize, device: &Device) -> Self {
        Self { n, inner: device.clone() }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl BurnSpace for DiscreteSpace<crate::core::Burn> {
    type Element = Tensor<1, Int>;

    fn sample(&self, n: usize) -> Self::Element {
        Tensor::random([n], burn::tensor::Distribution::Uniform(0., self.n as f64), &self.inner)
    }
    
    fn device(&self) -> Device {
        self.inner.clone()
    }
}

impl DiscreteSpace<crate::core::Native> {
    /// create a new `DiscreteSpace` with `Native` representation
    pub fn native(n: usize, seed: u64) -> Self {
        Self { n, inner: SmallRng::seed_from_u64(seed) }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl NativeSpace for DiscreteSpace<crate::core::Native> {
    type Element = usize;

    fn sample(&mut self) -> Self::Element {
        self.inner.random_range(0..self.n)
    }
}