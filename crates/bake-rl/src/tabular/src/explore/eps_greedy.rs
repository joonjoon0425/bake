//! Epsilon Greedy exploration strategy
//! 
use rand::{RngExt, SeedableRng};
use rand::rngs::SmallRng;

use crate::explore::Exploration;
use crate::constraint::Constraint;
use crate::qtable::{QTable, QValues};
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
    fn sample<C: Constraint>(&mut self, qtable: &QTable, obs: usize, constraint: C) -> usize {
        if self.rng.random_range(0.0..1.0) < self.eps {
            let n = constraint.n_possible_actions();
            let k = self.rng.random_range(0..n);
            constraint.iter().filter_map(|(a, p)| if *p { Some(a) } else { None }).nth(k).unwrap()
        } else {
            let values = qtable.qvalues(obs);
            let indices = values.argmaxes(constraint);
            let random = self.rng.random_range(0..indices.len());
            indices[random]
        }
    }

    fn prob<C: Constraint>(&self, qtable: &QTable, obs: usize, action: usize, constraint: C) -> f32 {
        let qvalues = qtable.qvalues(obs);
        let indices = qvalues.argmaxes(constraint);
        if indices.contains(&action) { (1. - self.eps) / indices.len() as f32 + self.eps / constraint.n_possible_actions() as f32 } else { self.eps / constraint.n_possible_actions() as f32 }
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint::DiscreteMask, explore::{EpsGreedy, Exploration}, qtable::QTable};

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
        let qtable = QTable::new(1, 4);

        for _ in 0..1000 {
            let action = e.sample(&qtable, 0, c.clone());
            assert!(action != 3);
        }
    }
}