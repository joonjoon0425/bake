//! Action mask trait and implementations for masking

/// Basic Mask trait which all masks must implement
pub trait Mask: Clone + Copy {
    /// returns true if given action is possible, else returns false
    fn is_possible(&self, action: usize) -> bool;
    /// returns the iterator of possible actions
    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_;
    /// returns the number of possible actions
    fn n_possible_actions(&self) -> usize;
    /// returns the number of all actions
    fn n_actions(&self) -> usize;
}

/// Basic discrete action mask implementation
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>([bool; D]);

/// struct used to represent a non-mask environment
#[derive(Debug, Clone, Copy)]
pub struct NoMask<const D: usize>;

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
}

impl<const D: usize> Mask for DiscreteMask<D> {
    fn is_possible(&self, action: usize) -> bool {
        self.0[action]
    }

    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate()
            .filter(|(_, possible)| **possible )
            .map(|(action, _)| action )
    }

    fn n_possible_actions(&self) -> usize {
        self.0.iter().filter(|possible| **possible ).count()
    }

    fn n_actions(&self) -> usize {
        D
    }
}

impl<const D: usize> Mask for NoMask<D> {
    fn is_possible(&self, _: usize) -> bool { true }

    fn possible_actions(&self) -> impl Iterator<Item = usize> {
        0..D
    }

    fn n_possible_actions(&self) -> usize {
        D
    }

    fn n_actions(&self) -> usize {
        D
    }
}