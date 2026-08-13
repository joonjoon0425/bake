use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{policy::argmaxes, types::*};

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

    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
    
    pub fn sample<M: Mask>(&mut self, qvalues: &[f32], mask: M) -> usize {
        let r = self.rng.random_range(0.0..1.0);

        let action = if r < self.eps {
            let k =self.rng.random_range(0..mask.n_possible_actions());
            mask.possible_actions().nth(k).unwrap()
        } else {
            let mut qmax = f32::MIN;
            let mut candidate = 0;
            let mut n_candidates = 1;
            for i in mask.possible_actions() {
                if qvalues[i] > qmax {
                    n_candidates = 1;
                    candidate = i;
                    qmax = qvalues[i];
                } else if (qvalues[i] - qmax).abs() < 1e-10 {
                    n_candidates += 1;
                    let r = self.rng.random_range(0..n_candidates);
                    if r == 0 { candidate = i }
                }
            }
            candidate
        };

        action
    }
    
    pub fn prob<M: Mask>(&self, qvalues: &[f32], action: usize, mask: M) -> f32 {
        let n_possible_actions = mask.n_possible_actions();
        let max_actions = argmaxes(qvalues, mask);

        if max_actions[action] {
            return self.eps / n_possible_actions as f32 + (1f32 - self.eps) / max_actions.iter().filter(|possible| **possible).count() as f32;
        } else {
            return self.eps / n_possible_actions as f32
        }
    }
}