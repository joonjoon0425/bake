//! Space for discrete elements

use crate::core::{Repr, env::space::Space};

/// Discrete space
#[derive(Debug, Clone)]
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

impl<M: Repr> Space<M> for DiscreteSpace {
    type Element = M::Index;
}