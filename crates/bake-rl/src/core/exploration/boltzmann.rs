//! boltzmann exploration
//! 

use crate::core::{constraint::discrete::DiscreteConstraint};
use burn::{prelude::*, tensor::activation::softmax};

use crate::deep::contract::DiscreteActionValue;
use super::Exploration;

/// Boltzmann (softmax) policy implementation
pub struct Boltzmann {
    temp: f32,
}

impl Boltzmann {
    /// create a new `Boltzmann` policy. The higher the temperature, more uniform-like the policy.
    pub fn new(temp: f32) -> Self {
        if temp < 0.0 { panic!("the temp must be bigger than 0.0. Current temp is {}", temp) } 
        Self {
            temp,
        }
    }

    /// get current temperature of Boltzmann policy
    pub fn temp(&self) -> f32 { self.temp }

    /// get the mutable reference of temperature of Boltzmann policy
    pub fn temp_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl Exploration for Boltzmann {
    fn sample<Q: DiscreteActionValue>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let value = qfunc.action_values(obs, constraint);
        let probs = softmax(value / self.temp, 1);
        let actions = probs.categorical(1).squeeze_dim(1);
        actions
    }
}