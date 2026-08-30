//! A wrapper which produces Constrained QNetwork from custom dueling qnet
//! 
use burn::prelude::*;
use crate::{approximator::QFunction, constraint::DiscreteConstraint, network::DuelingQNet};

/// A wrapper which produces Constrained QNetwork from custom dueling qnet
#[derive(Module, Debug)]
pub struct ConstrainedDuelingQNet<T: DuelingQNet> { net: T }

impl<T: DuelingQNet> ConstrainedDuelingQNet<T> {
    /// create a new Constrained Dueling QNet
    pub fn new(net: T) -> Self { Self { net } }
}

impl<T: DuelingQNet> QFunction for ConstrainedDuelingQNet<T> {
    type Obs = T::Obs;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let (value, advantage) = self.net.forward(obs);
        let mean = constraint.clone().mean_dim(1, advantage.clone());
        constraint.apply(value.unsqueeze_dim(1) + advantage - mean, -1e9)
    }
}