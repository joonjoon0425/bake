//! Action constraints for tabular rl

/// Basic constraint trait
pub trait Constraint: Clone + Copy {
    /// return the number of possible actions
    fn n_possible_actions(&self) -> usize;
    
    /// returns the iterator of possible actions
    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_;

    /// return true if given action is possible; else return false
    fn is_possible(&self, action: usize) -> bool;
}

/// Discrete constraint
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>([bool; D]);

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
    fn n_possible_actions(&self) -> usize {
        self.0.iter().filter(|&&a| a).count()
    }

    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().filter(|(_, p)| **p).map(|(a, _)| a)
    }

    fn is_possible(&self, action: usize) -> bool {
        assert!(0 <= action && action < D);
        self.0[action]
    }
}

/// No constraint
#[derive(Debug, Clone, Copy)]
pub struct Unconstrained<const D: usize>;
impl<const D: usize> Constraint for Unconstrained<D> {
    fn n_possible_actions(&self) -> usize { D }
    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_ { 0..D }
    fn is_possible(&self, _: usize) -> bool { true }
}

#[cfg(test)]
mod tests {
    use crate::constraint::{Constraint, DiscreteMask};

    #[test]
    fn n_possible_actions_test() {
        let c = DiscreteMask::from_bool([true, false, false, true]);
        assert!(c.n_possible_actions() == 2);
    }

    #[test]
    fn possible_actions_test() {
        let c = DiscreteMask::from_bool([true, true, false, true, false, true]);
        for a in c.possible_actions() {
            assert!(a != 2 && a != 4)
        }
    }

    #[test]
    #[should_panic]
    fn invalid_action_test() {
        let c = DiscreteMask::from_bool([true, true, false, false, true]);
        c.is_possible(5);
    }
}