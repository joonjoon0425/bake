//! The action value trait

use burn::prelude::*;
use crate::{constraint::discrete_constraint::DiscreteConstraint, experimental::contract::basic::network::Network};

/// The action value trait.
/// - The network which implements this trait is able to produce the action values, for discrete actions
pub trait DiscreteActionValue: Network {
    /// returns the action values of given observations
    /// - The constraint is applied
    fn action_values<C: DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> Tensor<2>;
}