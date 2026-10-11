//! Greedy policy (no exploration, only exploitation)
use crate::explore::Exploration;
use bake_rl_core::constraint::Constraint;
use crate::qtable::QValues;

/// Greedy policy implementation
/// - Unlike the bake-deep's Greedy, tabular version does the tie-break
pub struct Greedy;

impl Exploration for Greedy {
    fn sample<C: Constraint>(&mut self, qvalues: Vec<f32>, constraint: C) -> usize {
        let qvalues = constraint.apply(qvalues);
        let index = qvalues.argmax();
        index
    }

    fn prob<C: Constraint>(&self, qvalues: Vec<f32>, action: usize, constraint: C) -> f32 {
        let qvalues = constraint.apply(qvalues);
        let index = qvalues.argmax();
        if index == action && constraint.is_possible(action) { 1.0 } else { 0.0 }
    }
}