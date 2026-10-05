//! The trait for actor-critic methods
use burn::prelude::*;
use crate::{distribution::PossibleConstraint, contract::basic::{policy::Policy, state_value::StateValue}};

/// The special trait for actor-critic methods
/// - This trait is useful for encoder-sharing actor-critic methods; When the algorithm requires the actor and critic values at the same time, the function `actor_critic` can be used.
pub trait ActorCritic: Policy + StateValue {
    /// returns the distribution and state value
    fn dist_and_state_value<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, Tensor<1>);
}