//! epsilon-greedy policy

use burn::prelude::*;
use rand::prelude::*;
use crate::core::constraint::{BatchedDiscreteConstraint, DiscreteConstraint};

use super::DiscreteActionExploration;
use super::BatchedDiscreteActionExploration;

/// An epsilon-greedy policy implementation
pub struct EpsilonGreedy {
    eps: f32,
}

impl EpsilonGreedy {
    /// Create a new `EpsGreedy` policy. The epsilon is the probability of choosing action randomly.
    pub fn new(eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the epsilon must be between 0.0 and 1.0. Current epsilon is {}", eps) }
        Self {
            eps,
        }
    }
    /// get the value of epsilon
    pub fn epsilon(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn epsilon_mut(&mut self) -> &mut f32 { &mut self.eps }
}


impl DiscreteActionExploration for EpsilonGreedy {
    fn sample<C: DiscreteConstraint, R: Rng>(&self, rng: &mut R, qvalues: &[f32], constraint: C) -> usize {
        let qvalues = constraint.apply(qvalues, -1e9);
        let r: f32 = rng.random_range(0.0..1.0);
        if r < self.eps {
            let n = qvalues.len();
            let random: Vec<f32> = (0..n).into_iter().map(|_| rng.random()).collect();
            let rvalues = constraint.apply(&random, -1e9);
            super::argmax_max(&rvalues).0
        } else {
            super::argmax_max(&qvalues).0
        }
    }
}

/// An epsilon-greedy policy implementation
pub struct BatchedEpsilonGreedy {
    eps: f32,
}

impl BatchedEpsilonGreedy {
    /// Create a new `BatchedEpsGreedy` policy. The epsilon is the probability of choosing action randomly.
    pub fn new(eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the epsilon must be between 0.0 and 1.0. Current epsilon is {}", eps) }
        Self {
            eps,
        }
    }
    /// get the value of epsilon
    pub fn epsilon(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn epsilon_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl BatchedDiscreteActionExploration for BatchedEpsilonGreedy {
    fn sample<C: BatchedDiscreteConstraint>(&self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let n = qvalues.dims()[0];
        let r = Tensor::<1>::random([n], burn::tensor::Distribution::Uniform(0., 1.), &qvalues.device()).lower_elem(self.eps);
        let random_a = constraint.clone().apply(Tensor::random_like(&qvalues, burn::tensor::Distribution::Default), -1e9).argmax(1).squeeze_dim(1);
        let qvalues = constraint.apply(qvalues, -1e9).argmax(1).squeeze_dim(1);
        qvalues.mask_where(r, random_a)
    }
}
