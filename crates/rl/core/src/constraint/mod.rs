//! Action constraints for tabular rl

#[cfg(feature = "deep")]
use crate::deep::constraint::convert::ToBatchable;

/// Basic constraint trait
pub trait Constraint: Clone + Copy {
    /// apply the mask and return the result
    fn apply(&self, qvalue: Vec<f32>) -> Vec<f32>;
    /// return the number of possible actions
    fn n_possible_actions(&self) -> Option<usize>;
}

/// Discrete constraint
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>(pub [bool; D]);

impl<const D: usize> DiscreteMask<D> {
    /// create a new DiscreteMask as [enabled; D]
    pub fn new(enabled: bool) -> Self {
        Self([enabled; D])
    }
    /// create a given bool array into DiscreteMask
    pub fn from_bool(arr: [bool; D]) -> DiscreteMask<D> {
        Self(arr)
    }
    /// enable an action of given index
    pub fn enable(&mut self, idx: usize) {
        self.0[idx] = true;
    }
    /// disable an action of given index
    pub fn disable(&mut self, idx: usize) {
        self.0[idx] = false;
    }
}

impl<const D: usize> Constraint for DiscreteMask<D> {
    fn apply(&self, mut qvalue: Vec<f32>) -> Vec<f32> {
        assert!(qvalue.len() == D, "The number of action and the length of given qvalue does not matches");
        for i in 0..D {
            qvalue[i] = -1e9;
        }
        qvalue
    }

    fn n_possible_actions(&self) -> Option<usize> {
        Some(self.0.iter().filter(|p| **p).count())
    }
}

#[cfg(feature = "deep")]
impl<const D: usize> ToBatchable for DiscreteMask<D> {
    type Batched = crate::deep::constraint::discrete_constraint::DiscreteMask<2>;

    fn to_batchable(self, device: &burn::prelude::Device) -> Self::Batched {
        let data: Vec<bool> = self.0.into_iter().collect();
        crate::deep::constraint::discrete_constraint::DiscreteMask::<2>(burn::prelude::Tensor::<1, burn::prelude::Bool>::from_bool(data.as_slice(), device).unsqueeze_dim(0))
    }
}

/// No constraint
#[derive(Debug, Clone, Copy)]
pub struct Unconstrained;
impl Constraint for Unconstrained {
    fn apply(&self, qvalue: Vec<f32>) -> Vec<f32> { qvalue }
    fn n_possible_actions(&self) -> Option<usize> { None }
}

#[cfg(feature = "deep")]
impl ToBatchable for Unconstrained {
    type Batched = crate::deep::constraint::Unconstrained;

    fn to_batchable(self, _: &burn::prelude::Device) -> Self::Batched {
        crate::deep::constraint::Unconstrained
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::{Constraint, DiscreteMask};

    #[test]
    fn n_possible_actions_test() {
        let qvalue = vec![1., -1., 4., -5.];
        let c = DiscreteMask::from_bool([true, false, false, true]);
        let qvalue = c.apply(qvalue);
        assert!(qvalue[1] == -1e9);
        assert!(qvalue[3] == -1e9);
    }

    #[test]
    #[should_panic]
    fn invalid_length_test() {
        let qvalue = vec![1.];
        let c = DiscreteMask::from_bool([true, true, false, false, true]);
        c.apply(qvalue);
    }
}