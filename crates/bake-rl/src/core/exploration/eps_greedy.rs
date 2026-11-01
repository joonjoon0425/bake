//! epsilon-greedy policy
//!
use crate::core::{constraint::discrete::DiscreteConstraint};
use burn::{Tensor, tensor::{Distribution, Int}};
use super::Exploration;

/// An epsilon-greedy policy implementation
pub struct EpsGreedy {
    eps: f32,
}

impl EpsGreedy {
    /// Create a new `EpsGreedy` policy. The epsilon is the probability of choosing action randomly
    pub fn new(eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the eps must be between 0.0 and 1.0. Current eps is {}", eps) }
        Self {
            eps,
        }
    }

    /// get the value of epsilon
    pub fn eps(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl Exploration for EpsGreedy {
    fn sample<C: DiscreteConstraint>(&mut self, qvalues: Tensor<2>, constraint: C) -> Tensor<1, Int> {
        let n = qvalues.dims()[0];
        let r = Tensor::<1>::random([n], Distribution::Uniform(0., 1.), &qvalues.device()).lower_elem(self.eps);
        let random_a = constraint.clone().apply(Tensor::random_like(&qvalues, Distribution::Default), -1e9).argmax(1).squeeze_dim(1);
        let qvalues = constraint.apply(qvalues, -1e9).argmax(1).squeeze_dim(1);
        qvalues.mask_where(r, random_a)
    }
}