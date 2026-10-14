//! Continuous space trait
//! 

use crate::core::env::space::Space;

/// A continuous space
pub trait ContinuousSpace: Space {
    /// return the shape
    fn shape(&self) -> &[usize];
    /// return the lower bound
    fn low(&self) -> &<Self as Space>::Element;
    /// return the upper bound
    fn high(&self) -> &<Self as Space>::Element;
}