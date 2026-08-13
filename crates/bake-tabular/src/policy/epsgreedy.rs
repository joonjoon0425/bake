use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::types::ActionMask;

pub struct EpsGreedy {
    eps: f32,
    rng: StdRng,
}

impl EpsGreedy {
    pub fn new(seed: u64, eps: f32) -> Self {
        Self {
            eps,
            rng: StdRng::seed_from_u64(seed),
        }
    }
    // no tie-break for simplicity
    pub fn sample(&mut self, qvalues: &[f32], mask: Option<ActionMask>) -> usize {
        let r = self.rng.random_range(0.0..1.0);

        let action = if r < self.eps {
            
        } else {
            mask.un
        };

        action
    }
    // no tie-break for simplicity
    pub fn prob(&self, qvalues: &[f32], mask: Option<ActionMask>) -> f32 {

    }
}