//! boltzmann exploration
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use crate::core::constraint::{BatchedDiscreteConstraint, DiscreteConstraint};

use super::DiscreteActionExploration;
use super::BatchedDiscreteActionExploration;

/// Boltzmann (softmax) policy implementation
pub struct Boltzmann {
    temp: f32
}

impl Boltzmann {
    /// create a new `Boltzmann` policy. The higher the temperature, more uniform-like the policy.
    pub fn new(temp: f32) -> Self {
        if temp <= 0.0 { panic!("the temperature must be greater than 0.0. Current temperature is {}", temp) } 
        Self {
            temp,
        }
    }

    /// get current temperature of Boltzmann policy
    pub fn temperature(&self) -> f32 { self.temp }

    /// get the mutable reference of temperature of Boltzmann policy
    pub fn temperature_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl DiscreteActionExploration for Boltzmann {
    fn sample<C: DiscreteConstraint, R: Rng>(&self, rng: &mut R, qvalues: &[f32], constraint: C) -> usize {
        let mut qvalues = constraint.apply(qvalues, -1e9);
        let vmax = super::argmax_max(&qvalues).1;
        for qvalue in &mut qvalues {
            *qvalue = ((*qvalue - vmax) / self.temp).exp();
        }
        let weighted_index = WeightedIndex::new(qvalues).unwrap();
        weighted_index.sample(rng)
    }
}

/// Boltzmann (softmax) policy implementation
pub struct BatchedBoltzmann {
    temp: f32
}

impl BatchedBoltzmann {
    /// create a new `Boltzmann` policy. The higher the temperature, more uniform-like the policy.
    pub fn burn(temp: f32) -> Self {
        if temp <= 0.0 { panic!("the temperature must be greater than 0.0. Current temperature is {}", temp) } 
        Self {
            temp,
        }
    }

    /// get current temperature of Boltzmann policy
    pub fn temperature(&self) -> f32 { self.temp }

    /// get the mutable reference of temperature of Boltzmann policy
    pub fn temperature_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl BatchedDiscreteActionExploration for BatchedBoltzmann {
    fn sample<C: BatchedDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let qvalues = constraint.apply(qvalues, -1e9);
        let probs = softmax(qvalues / self.temp, 1);
        let actions = probs.categorical(1).squeeze_dim(1);
        actions
    }
}

