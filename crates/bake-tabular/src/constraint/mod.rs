//! Action constraints for tabular rl

/// Basic constraint trait
pub trait Constraint: Clone {
    /// return the number of possible actions
    fn n_possible_actions(&self) -> usize;
    /// apply the mask to given values
    fn apply(&self, values: &mut [f32]);
}

/// Discrete constraint
#[derive(Debug, Clone)]
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

    fn apply(&self, values: &mut [f32]) {
        if values.len() != D { panic!("The expected length of given values as {}, recieved {}.", D, values.len()) }
        for (i, &p) in self.0.iter().enumerate() {
            if !p { values[i] = -1e9 }
        }
    }
}

/// No constraint
#[derive(Debug, Clone, Copy)]
pub struct Unconstrained<const D: usize>;
impl<const D: usize> Constraint for Unconstrained<D> {
    fn n_possible_actions(&self) -> usize { D }
    fn apply(&self, _: &mut [f32]) { }
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
    fn apply_test() {
        let mut values = [1.0, 2.0, 31.0, -1.0, -33.0, 15.0];
        let c = DiscreteMask::from_bool([true, true, false, true, false, true]);
        c.apply(&mut values);
        assert!(values[0] == 1.0);
        assert!(values[1] == 2.0);
        assert!(values[2] == -1e9);
        assert!(values[3] == -1.0);
        assert!(values[4] == -1e9);
        assert!(values[5] == 15.0);
    }

    #[test]
    #[should_panic]
    fn action_num_mismatch_test() {
        let c = DiscreteMask::from_bool([true, true, false, false, true]);
        let mut values = [0.1, 0.2, 0.3];
        c.apply(&mut values);
    }
}