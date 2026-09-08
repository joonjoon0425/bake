//! Greedy policy (no exploration, only exploitation)
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::explore::Exploration;
use crate::constraint::Constraint;
use crate::qtable::{QTable, QValues};

/// Greedy policy implementation
/// - Unlike the bake-deep's Greedy, tabular version does the tie-break
pub struct Greedy {
    rng: SmallRng,
}

impl Greedy {
    /// create a new Greedy policy
    pub fn new(seed: u64) -> Self { Self { rng: SmallRng::seed_from_u64(seed) } }
}

impl Exploration for Greedy {
    fn sample<C: Constraint>(&mut self, qtable: &QTable, obs: usize, constraint: C) -> usize {
        let mut qvalues = qtable.qvalues_as_vec(obs);
        constraint.apply(&mut qvalues);
        let indices = qvalues.argmaxes();
        let random = self.rng.random_range(0..indices.len());
        indices[random]
    }

    fn prob<C: Constraint>(&self, qtable: &QTable, obs: usize, action: usize, constraint: C) -> f32 {
        let mut qvalues = qtable.qvalues_as_vec(obs);
        constraint.apply(&mut qvalues);
        let indices = qvalues.argmaxes();
        if indices.contains(&action) { 1.0 / indices.len() as f32 } else { 0.0 }
    }
}