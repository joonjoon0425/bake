//! Discrete Action Value

use crate::core::constraint::BatchedDiscreteConstraint;
use crate::core::action_selector::BatchedDiscreteActionSelector;
use crate::deep::network::Network;
use burn::prelude::*;
/// The discrete action value trait.
/// - The network which implements this trait is able to produce the action values, for discrete actions
pub trait DiscreteActionValue: Network {
    /// returns the action values of given observations
    /// - The constraint is applied
    fn action_values<C: BatchedDiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> Tensor<2>;
    
    /// sample an action from given exploration strategy
    fn action<A: BatchedDiscreteActionSelector, C: BatchedDiscreteConstraint>(&self, action_selector: &mut A, obs: Self::Obs, constraint: C) -> Tensor<1, Int> {
        let qvalues = self.action_values(obs, constraint.clone());
        action_selector.sample(qvalues, constraint)
    }
}