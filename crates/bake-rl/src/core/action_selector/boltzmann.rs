//! boltzmann exploration
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use crate::core::action_selector::ActionSelectorRepr;
use crate::core::constraint::{BurnDiscreteConstraint, NativeDiscreteConstraint};

use super::NativeDiscreteActionSelector;
use super::BurnDiscreteActionSelector;

/// Boltzmann (softmax) policy implementation
pub struct Boltzmann<Repr: ActionSelectorRepr> {
    temp: f32,
    inner: Repr::Inner,
}

impl<Repr: ActionSelectorRepr> Boltzmann<Repr> {
    /// get current temperature of Boltzmann policy
    pub fn temperature(&self) -> f32 { self.temp }

    /// get the mutable reference of temperature of Boltzmann policy
    pub fn temperature_mut(&mut self) -> &mut f32 { &mut self.temp }
}

/// `Burn` repr
impl Boltzmann<crate::core::Burn> {
    /// create a new `Boltzmann` policy with `Burn` representation. The higher the temperature, more uniform-like the policy.
    pub fn burn(temp: f32) -> Self {
        if temp < 0.0 { panic!("the temperature must be bigger than 0.0. Current temperature is {}", temp) } 
        Self {
            temp,
            inner: ()
        }
    }
}

impl BurnDiscreteActionSelector for Boltzmann<crate::core::Burn> {
    fn sample<C: BurnDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let qvalues = constraint.apply(qvalues, -1e9);
        let probs = softmax(qvalues / self.temp, 1);
        let actions = probs.categorical(1).squeeze_dim(1);
        actions
    }
}

/// `Native` repr
impl Boltzmann<crate::core::Native> {
    /// create a new `Boltzmann` policy with `Native` representation. The higher the temperature, more uniform-like the policy.
    pub fn native(temp: f32, seed: u64) -> Self {
        if temp < 0.0 { panic!("the temperature must be bigger than 0.0. Current temperature is {}", temp) } 
        Self {
            temp,
            inner: SmallRng::seed_from_u64(seed)
        }
    }
}

impl NativeDiscreteActionSelector for Boltzmann<crate::core::Native> {
    fn sample<C: NativeDiscreteConstraint>(&mut self, qvalues: &[f32], constraint: C) -> usize {
        let mut qvalues = constraint.apply(qvalues, -1e9);
        let vmax = super::argmax_max(&qvalues).1;
        for qvalue in &mut qvalues {
            *qvalue = ((*qvalue - vmax) / self.temp).exp();
        }
        let weighted_index = WeightedIndex::new(qvalues).unwrap();
        weighted_index.sample(&mut self.inner)
    }
}