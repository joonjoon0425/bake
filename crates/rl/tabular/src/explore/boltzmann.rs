//! boltzmann exploration strategy
//! 
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::explore::Exploration;
use bake_rl_core::constraint::Constraint;
use crate::qtable::QValues;
/// boltzmann exploration strategy implementation
/// - chooses action using q values as logits for categorical distribution
pub struct Boltzmann {
    temp: f32,
    rng: SmallRng
}

impl Boltzmann {
    /// create a new boltzmann policy
    pub fn new(seed: u64, temp: f32) -> Self {
        if temp <= 0.0 { panic!("the temp must be bigger than 0.0. Current temp is {}", temp) } 
        Self { temp, rng: SmallRng::seed_from_u64(seed) }
    }

    /// give current temperature
    pub fn temp(&self) -> f32 { self.temp }

    /// give a mutable reference of temperature
    pub fn temp_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl Exploration for Boltzmann {
    fn sample<C: Constraint>(&mut self, qvalues: Vec<f32>, constraint: C) -> usize {
        let mut qvalues = constraint.apply(qvalues);
        let vmax = qvalues.max();
        for qvalue in &mut qvalues {
            *qvalue = ((*qvalue - vmax) / self.temp).exp();
        }
        let weighted_index = WeightedIndex::new(qvalues).unwrap();
        weighted_index.sample(&mut self.rng)
    }

    fn prob<C: Constraint>(&self, qvalues: Vec<f32>,action: usize, constraint: C) -> f32 {
        let qvalues = constraint.apply(qvalues);
        let vmax = qvalues.max();
        let numerator = ((qvalues[action] - vmax) / self.temp).exp();
        let mut denominator = 0f32;
        for qvalue in qvalues {
            denominator += ((qvalue - vmax) / self.temp).exp();
        }
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use crate::{explore::{Boltzmann, Exploration}, qtable::QTableConfig};
    use bake_rl_core::constraint::DiscreteMask;

    #[test]
    #[should_panic]
    fn wrong_temp_below_zero() {
        Boltzmann::new(0, -2.0);
    }

    #[test]
    fn constraint_test() {
        let c = DiscreteMask::from_bool([true, true, true, false]);
        let mut e = Boltzmann::new(1, 1000f32);
        let qtable = QTableConfig::new(1, 4).init();

        for _ in 0..1000 {
            let action = e.sample(qtable.qvalues_as_vec(0), c.clone());
            assert!(action != 3);
        }
    }
}