//! exploration traits and implementations
use burn::prelude::*;
use rand::prelude::*;
use crate::core::constraint::{BurnDiscreteConstraint, NativeDiscreteConstraint};
/// A trait which all native representation exploration strategies must implement
pub trait NativeDiscreteActionSelector {
    /// sample an action using given discrete action values, with given constraint
    fn sample<C: NativeDiscreteConstraint>(&mut self, qvalues: &[f32], constraint: C) -> usize;
}

/// A trait which all native representation exploration strategies must implement
pub trait BurnDiscreteActionSelector {
    /// sample an action using given discrete action values, with given constraint
    fn sample<C: BurnDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int>;
}

/// Action selector representation
pub trait ActionSelectorRepr {
    /// inner types which each representation requires
    type Inner;
}
impl ActionSelectorRepr for crate::core::Native { type Inner = SmallRng; }
impl ActionSelectorRepr for crate::core::Burn { type Inner = (); }

pub mod greedy;
pub use greedy::Greedy;

pub mod eps_greedy;
pub use eps_greedy::EpsGreedy;

pub mod boltzmann;
pub use boltzmann::Boltzmann;

/// utility max function with index
fn argmax_max(values: &[f32]) -> (usize, f32) {
    let mut max = f32::MIN;
    let mut idx = 0;

    for (i, &v) in values.iter().enumerate() {
        if max < v {
            max = v;
            idx = i;
        }
    }
    (idx, max)
}