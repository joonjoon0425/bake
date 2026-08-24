//! Network structure for actor-critic method

use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::{distribution::Distribution, types::Batchable};

/// ActorCritic trait which all actor-critic algorithms require
pub trait ActorCritic : AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;
    /// the distribution which actor produces
    type Dist: Distribution;
    /// the constraint associated with observation
    type Constraint: Batchable;

    /// returns the distribution object according to current obs and constraint
    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist;
    /// return the state value according to given obs
    fn critic(&self, obs: Self::Obs) -> Tensor<1>;

    /// For the encoder-sharing network, this function must be overloaded appropriately
    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> (Self::Dist, Tensor<1>) {
        (self.actor(obs.clone(), constraint), self.critic(obs))
    }

    /// sample an action from actor
    fn action(&self, obs: Self::Obs, constraint: Self::Constraint) -> <Self::Dist as Distribution>::Action {
        self.actor(obs, constraint).sample()
    }

    /// checks whether this actor-critic shares encoder
    fn shares_encoder(&self) -> bool;
}
