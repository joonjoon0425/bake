//! Discrete Space trait and its implementations

use crate::space::Space;

/// usize typed Discrete Space
pub struct DiscreteSpace { n: usize }
impl DiscreteSpace {
    pub fn new(n: usize) -> Self { Self { n } }
    pub fn n(&self) -> usize {
        self.n
    }
}

impl Space for DiscreteSpace { type Element = usize; }