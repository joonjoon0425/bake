//! Greedy policy implementation
use crate::{policy::Policy, types::*};

/// Implementation of greedy policy. No tie-breaking here
pub struct Greedy;

impl Policy for Greedy {
    fn sample<M: Mask>(&mut self, qvalues: &[f32], mask: M) -> usize {
        let mut qmax = f32::MIN;
        let mut candidate = 0;
        for i in mask.possible_actions() {
            if qvalues[i] > qmax {
                candidate = i;
                qmax = qvalues[i];
            }
        }
        candidate
    }
    
    fn prob<M: Mask>(&self, qvalues: &[f32], action: usize, mask: M) -> f32 {
        let mut qmax = f32::MIN;
        let mut max_action = 0;
        for i in mask.possible_actions() {
            if qvalues[i] > qmax {
                max_action = i;
                qmax = qvalues[i];
            }
        }

        if max_action == action {
            1f32
        } else {
            0f32
        }
    }
}