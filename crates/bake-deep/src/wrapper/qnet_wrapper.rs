//! A wrapper which produces Constrained QNetwork from custom qnet
//! 
use burn::prelude::*;

use crate::{constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction, net::{DiscreteDuelingQNet, DiscreteQNet}};
/// A wrapper which produces Constrained QNetwork from custom qnet
#[derive(Module, Debug)]
pub struct DiscreteQNetWrapper<T: DiscreteQNet> { net: T }

impl<T: DiscreteQNet> DiscreteQNetWrapper<T> {
    /// create a new DiscreteQFuction from given DiscreteQNet
    pub fn new(net: T) -> Self { Self { net } }
}

// impl<T: QNet + NoiseReset> NoiseReset for ConstrainedQNet<T> {
//     fn reset_noise(&mut self) {
//         self.net.reset_noise();
//     }
// }

impl<T: DiscreteQNet> DiscreteQFunction for DiscreteQNetWrapper<T> {
    type Obs = T::Obs;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2> {
        constraint.apply(self.net.forward(obs), -1e9)
    }
}

/// A wrapper which produces DiscreteQFunction from custom dueling qnet
#[derive(Module, Debug)]
pub struct DiscreteDuelingQNetWrapper<T: DiscreteDuelingQNet> { net: T }

impl<T: DiscreteDuelingQNet> DiscreteDuelingQNetWrapper<T> {
    /// create a new DiscreteQFunction from given DiscreteDuelingQNet
    pub fn new(net: T) -> Self { Self { net } }
}

// impl<T: DuelingQNet + NoiseReset> NoiseReset for ConstrainedDuelingQNet<T> { fn reset_noise(&mut self) { self.net.reset_noise(); } }

impl<T: DiscreteDuelingQNet> DiscreteQFunction for DiscreteDuelingQNetWrapper<T> {
    type Obs = T::Obs;

    fn forward(&self, obs: Self::Obs, constraint: impl DiscreteConstraint) -> Tensor<2> {
        let (value, advantage) = self.net.forward(obs);
        let mean = constraint.clone().mean_dim(1, advantage.clone());
        constraint.apply(value.unsqueeze_dim(1) + advantage - mean, -1e9)
    }
}