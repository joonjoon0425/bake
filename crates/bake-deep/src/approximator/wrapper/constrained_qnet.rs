//! A wrapper which produces Constrained QNetwork from custom qnet
//! 
use burn::prelude::*;
use crate::{approximator::QFunction, constraint::DiscreteConstraint, network::QNet};

/// A wrapper which produces Constrained QNetwork from custom qnet
#[derive(Module, Debug)]
pub struct ConstrainedQNet<T: QNet> { net: T }

impl<T: QNet> ConstrainedQNet<T> {
    /// create a new Constrained QNet
    pub fn new(net: T) -> Self { Self { net } }
}

impl<T: QNet> QFunction for ConstrainedQNet<T> {
    type Obs = T::Obs;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2> {
        constraint.apply(self.net.forward(obs), -1e9)
    }
}