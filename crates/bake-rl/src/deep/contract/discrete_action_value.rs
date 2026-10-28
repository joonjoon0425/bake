//! Discrete Action Value

use crate::core::constraint::DiscreteConstraint;
use crate::core::data::BatchTensor;
use crate::deep::network::Network;
/// The discrete action value trait.
/// - The network which implements this trait is able to produce the action values, for discrete actions
pub trait DiscreteActionValue: Network {
    /// returns the action values of given observations
    /// - The constraint is applied
    fn action_values<C: DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> BatchTensor<2>;
}