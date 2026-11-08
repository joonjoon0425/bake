//! epsilon-greedy policy

use burn::prelude::*;
use rand::prelude::*;
use crate::core::action_selector::ActionSelectorRepr;
use crate::core::constraint::{BurnDiscreteConstraint, NativeDiscreteConstraint};

use super::NativeDiscreteActionSelector;
use super::BurnDiscreteActionSelector;

/// An epsilon-greedy policy implementation
pub struct EpsilonGreedy<Repr: ActionSelectorRepr> {
    eps: f32,
    inner: Repr::Inner,
}

/// common
impl<Repr: ActionSelectorRepr> EpsilonGreedy<Repr> {
    /// get the value of epsilon
    pub fn epsilon(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn epsilon_mut(&mut self) -> &mut f32 { &mut self.eps }
}

/// `Burn` repr
impl EpsilonGreedy<crate::core::Burn> {
    /// Create a new `EpsGreedy` policy with `Burn` representation. The epsilon is the probability of choosing action randomly.
    pub fn burn(eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the epsilon must be between 0.0 and 1.0. Current epsilon is {}", eps) }
        Self {
            eps,
            inner: (),
        }
    }
}

impl BurnDiscreteActionSelector for EpsilonGreedy<crate::core::Burn> {
    fn sample<C: BurnDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let n = qvalues.dims()[0];
        let r = Tensor::<1>::random([n], burn::tensor::Distribution::Uniform(0., 1.), &qvalues.device()).lower_elem(self.eps);
        let random_a = constraint.clone().apply(Tensor::random_like(&qvalues, burn::tensor::Distribution::Default), -1e9).argmax(1).squeeze_dim(1);
        let qvalues = constraint.apply(qvalues, -1e9).argmax(1).squeeze_dim(1);
        qvalues.mask_where(r, random_a)
    }
}

/// `Native` repr
impl EpsilonGreedy<crate::core::Native> {
    /// Create a new `EpsGreedy` policy with `Native` representation. The epsilon is the probability of choosing action randomly.
    pub fn native(eps: f32, seed: u64) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the epsilon must be between 0.0 and 1.0. Current epsilon is {}", eps) }
        Self {
            eps,
            inner: SmallRng::seed_from_u64(seed),
        }
    }
}

impl NativeDiscreteActionSelector for EpsilonGreedy<crate::core::Native> {
    fn sample<C: NativeDiscreteConstraint>(&mut self, qvalues: &[f32], constraint: C) -> usize {
        let qvalues = constraint.apply(qvalues, -1e9);
        let r: f32 = self.inner.random_range(0.0..1.0);
        if r < self.eps {
            let n = qvalues.len();
            let random: Vec<f32> = (0..n).into_iter().map(|_| self.inner.random()).collect();
            let rvalues = constraint.apply(&random, -1e9);
            super::argmax_max(&rvalues).0
        } else {
            super::argmax_max(&qvalues).0
        }
    }
}