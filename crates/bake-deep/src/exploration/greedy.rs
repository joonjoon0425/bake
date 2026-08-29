//! greedy policy implementation
use burn::prelude::*;
use crate::{approximator::QFunction, constraint::DiscreteConstraint, exploration::Exploration};

pub struct Greedy;

impl Exploration for Greedy {
    fn sample<QFunc: QFunction>(&mut self, qfunc: &QFunc, obs: QFunc::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = qfunc.forward(obs, constraint);
        qvalues.argmax(1).squeeze_dim(1)
    }

    fn prob<QFunc: QFunction>(&self, qfunc: &QFunc, obs: QFunc::Obs, action: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<1> {
        let qvalues = qfunc.forward(obs, constraint);
        let argmax = qvalues.argmax(1).squeeze_dim(1);
        argmax.equal(action).float()
    }
}