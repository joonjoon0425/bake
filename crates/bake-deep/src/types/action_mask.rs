//! Action mask trait and implementations for masking
use burn::Tensor;
use crate::types::Batchable;
/// Basic Mask trait which all masks must implement
pub trait ActionMask: Batchable {
    type Value;

    fn apply(batched_mask: <Self as Batchable>::Batched, values: Self::Value, fill_value: f32) -> Self::Value;
}
/// Discrete Action Mask struct
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>(pub [bool; D]);

impl<const D: usize> DiscreteMask<D> {
    /// create a new DiscreteMask as [enabled; D]
    pub fn new(enabled: bool) -> Self {
        Self ([enabled; D])
    }

    /// enable an action of given index
    pub fn enable(&mut self, idx: usize) {
        self.0[idx] = true;
    }

    /// disable an action of given index
    pub fn disable(&mut self, idx: usize) {
        self.0[idx] = false;
    }
    
    /// checks whether given action is possible
    pub fn is_possible(&self, action: usize) -> bool {
        self.0[action]
    }

    /// returns all possible actions as Iterator
    pub fn possible_actions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate()
            .filter(|(_, possible)| **possible )
            .map(|(action, _)| action )
    }

    /// returns the number of possible actions
    pub fn n_possible_actions(&self) -> usize {
        self.0.iter().filter(|possible| **possible ).count()
    }

    /// returns the number of total actions
    pub fn n_actions(&self) -> usize {
        D
    }
}

impl<const D: usize> ActionMask for DiscreteMask<D> {
    type Value = Tensor<2>;

    fn apply(batched_mask: <Self as Batchable>::Batched, value: Self::Value, fill_value: f32) -> Self::Value {
        value.mask_fill(batched_mask, fill_value)
    }
}

impl ActionMask for () {
    type Value = Tensor<2>;

    fn apply(_: <Self as Batchable>::Batched, values: Self::Value, _: f32) -> Self::Value { values }
}