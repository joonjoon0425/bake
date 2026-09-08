//! boltzmann exploration strategy
//! 
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::explore::Exploration;
use crate::constraint::Constraint;
use crate::qtable::QTable;
/// boltzmann exploration strategy implementation
/// - chooses action using q values as logits for categorical distribution
pub struct Boltzmann {
    temp: f32,
    rng: SmallRng
}

impl Boltzmann {
    /// create a new boltzmann policy
    pub fn new(seed: u64, temp: f32) -> Self {
        if temp < 0.0 { panic!("the temp must be bigger than 0.0. Current temp is {}", temp) } 
        Self { temp, rng: SmallRng::seed_from_u64(seed) }
    }

    /// give current temperature
    pub fn temp(&self) -> f32 { self.temp }

    /// give a mutable reference of temperature
    pub fn temp_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl Exploration for Boltzmann {
    fn sample<C: Constraint>(&mut self, qtable: &QTable, obs: usize, constraint: C) -> usize {
        let mut values = qtable.qvalues_as_vec(obs);
        constraint.apply(&mut values);
        for v in values.iter_mut() { *v = v.exp() / self.temp }
        let weighted_index = WeightedIndex::new(values).unwrap();
        weighted_index.sample(&mut self.rng)
    }

    fn prob<C: Constraint>(&self, qtable: &QTable, obs: usize, action: usize, constraint: C) -> f32 {
        let mut values = qtable.qvalues_as_vec(obs);
        constraint.apply(&mut values);
        for v in values.iter_mut() { *v = v.exp() / self.temp }
        let sum = values.iter().sum::<f32>();
        values[action] / sum
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint::DiscreteMask, explore::{Boltzmann, Exploration}, qtable::QTable};

    #[test]
    #[should_panic]
    fn wrong_temp_below_zero() {
        Boltzmann::new(0, -2.0);
    }

    #[test]
    fn mask_test() {
        let c = DiscreteMask::from_bool([true, false, false, false]);
        let mut e = Boltzmann::new(1, 1.0);
        let qtable = QTable::new(1, 4);

        for _ in 0..100 {
            let action = e.sample(&qtable, 0, c.clone());
            assert!(action == 0);
        }
    }

    #[test]
    fn constraint_test() {
        let c = DiscreteMask::from_bool([true, true, true, false]);
        let mut e = Boltzmann::new(1, 0.001);
        let qtable = QTable::new(1, 4);

        for _ in 0..1000 {
            let action = e.sample(&qtable, 0, c.clone());
            assert!(action != 3);
        }
    }
}