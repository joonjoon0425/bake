//! Discrete space trait
//! 

use crate::core::env::space::Space;

/// A discrete space
pub trait DiscreteSpace: Space {
    /// return the number of elements
    fn n(&self) -> usize;
}