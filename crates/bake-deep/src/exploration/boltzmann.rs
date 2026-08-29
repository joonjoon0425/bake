//! boltzmann exploration
//! 
use burn::{prelude::*, tensor::activation::softmax};
use crate::{approximator::QFunction, constraint::DiscreteConstraint, exploration::Exploration};

pub struct Boltzmann {
    temp: f32,
}

impl Boltzmann {
    pub fn new(temp: f32) -> Self {
        Self {
            temp,
        }
    }

    pub fn temp(&self) -> f32 { self.temp }
    pub fn temp_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl Exploration for Boltzmann {
    fn sample<QFunc: QFunction>(&mut self, qfunc: &QFunc, obs: QFunc::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let probs = softmax(qfunc.forward(obs, constraint) / self.temp, 1);
        let actions = probs.categorical(1).squeeze_dim(1);
        actions
    }

    fn prob<QFunc: QFunction>(&self, qfunc: &QFunc, obs: QFunc::Obs, action: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<1> {
        let probs = softmax(qfunc.forward(obs, constraint) / self.temp, 1);
        let probs = probs.gather(1, action.unsqueeze_dim(1)).squeeze_dim(1);
        probs
    }
}