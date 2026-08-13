//! # Policies for tabular algorithms
//! Exploration and exploitation strategies are implemented here

pub mod epsgreedy;
pub use epsgreedy::*;

use crate::types::Mask;

/// returns a vector of actions with maximum q values
pub fn argmaxes<M: Mask>(qvalues: &[f32], mask: M) -> Vec<bool> {
    let mut qmax = f32::MIN;
    let mut candidates = vec![false; mask.n_actions()];
    for i in mask.possible_actions() {
        if qvalues[i] > qmax {
            candidates.fill(false);
            candidates[i] = true;
            qmax = qvalues[i];
        } else if qvalues[i] - qmax < 1e-10 {
            candidates[i] = true;
        }
    }
    candidates
}

/// returns a maximum q value
pub fn max<M: Mask>(qvalues: &[f32], mask: M) -> f32 {
    let mut qmax = f32::MIN;
    for i in mask.possible_actions() {
        if qvalues[i] > qmax {
            qmax = qvalues[i];
        }
    }
    qmax
}