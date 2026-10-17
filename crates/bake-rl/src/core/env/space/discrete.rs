//! Space for discrete elements

use crate::core::env::space::Space;
use burn::prelude::*;

/// Discrete space
pub struct DiscreteSpace {
    n: usize,
}

impl DiscreteSpace {
    /// create a new `DiscreteSpace`
    pub fn new(n: usize) -> Self {
        Self { n }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl Space for DiscreteSpace {
    type Element = Tensor<1, Int>;
}