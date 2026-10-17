//! single discrete space

use crate::core::env::space::Space;

/// single discrete space
#[derive(Debug, Clone)]
pub struct SingleDiscreteSpace {
    n: usize,
}

impl SingleDiscreteSpace {
    /// create a new `DiscreteSpace`
    pub fn new(n: usize) -> Self {
        Self { n }
    }
    /// return the number of elements
    pub fn n(&self) -> usize { self.n }
}

impl Space for SingleDiscreteSpace {
    type Element = usize;
}