//! Epsilon Greedy exploration strategy
//! 
use rand::{RngExt, SeedableRng};
use rand::rngs::SmallRng;

use crate::explore::Exploration;
use bake_rl_core::constraint::Constraint;
use crate::qtable::QValues;
/// epsilon greedy exploration strategy implementation
/// - exploits with probability `1 - eps` and explores with probability `eps`,
pub struct EpsGreedy {
    eps: f32,
    rng: SmallRng
}

impl EpsGreedy {
    /// create a new Epsilon greedy policy
    pub fn new(seed: u64, eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the eps must be between 0.0 and 1.0. Current eps is {}", eps) } 
        Self { eps, rng: SmallRng::seed_from_u64(seed) }
    }

    /// give current eps
    pub fn eps(&self) -> f32 { self.eps }

    /// give a mutable reference of eps
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl Exploration for EpsGreedy {
    fn sample<C: Constraint>(&mut self, qvalues: Vec<f32>, constraint: C) -> usize {
        if self.rng.random_range(0.0..1.0) < self.eps {
            let n = qvalues.len();
            let r = (0..n).into_iter().map(|_| self.rng.random_range(0.0..1.0) ).collect();
            let r = constraint.apply(r);
            r.argmax()
        } else {
            let qvalues = constraint.apply(qvalues);
            let index = qvalues.argmax();
            index
        }
    }

    fn prob<C: Constraint>(&self, qvalues: Vec<f32>, action: usize, constraint: C) -> f32 {
        let qvalues = constraint.apply(qvalues);
        let index = qvalues.argmax();
        let n_possible_actions = constraint.n_possible_actions().unwrap_or_else(|| {
            qvalues.len()
        });
        if index == action && constraint.is_possible(action) { (1. - self.eps) + self.eps / n_possible_actions as f32 } else { self.eps / n_possible_actions as f32 }
    }
}

#[cfg(test)]
mod tests {
    use crate::{explore::{EpsGreedy, Exploration}, qtable::QTableConfig};
    use bake_rl_core::constraint::DiscreteMask;
    #[test]
    #[should_panic]
    fn wrong_eps_over_one() {
        EpsGreedy::new(0, 2.0);
    }

    #[test]
    #[should_panic]
    fn wrong_eps_below_zero() {
        EpsGreedy::new(0, -2.0);
    }

    #[test]
    fn constraint_test() {
        let c = DiscreteMask::from_bool([true, true, true, false]);
        let mut e = EpsGreedy::new(1, 1.0);
        let qtable = QTableConfig::new(1, 4).init();

        for _ in 0..1000 {
            let action = e.sample(qtable.qvalues_as_vec(0), c.clone());
            assert!(action != 3);
        }
    }
}