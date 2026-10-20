//! Space for discrete elements

use crate::core::{Repr, env::space::Space};

/// Discrete space
#[derive(Debug, Clone)]
pub struct DiscreteSpace<R: Repr> {
    n: R::Index,
}

impl<R: Repr> DiscreteSpace<R> {
    /// create a new `DiscreteSpace`
    pub fn new(n: R::Index) -> Self {
        Self { n }
    }
    /// return the number of elements
    pub fn n(&self) -> &R::Index { &self.n }
}

impl<M: Repr> Space<M> for DiscreteSpace<M> {
    type Element = M::Index;
}