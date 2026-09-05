//! A trait for actor critic methods 

use burn::{Tensor, module::{AutodiffModule, ModuleDisplay}};
use crate::{data::batchable::Batchable, distribution::{Distribution, PossibleConstraint}};

/// ActorCritic trait which all actor-critic algorithms require
pub trait ActorCritic: AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;
    /// the distribution which actor produces
    type Dist: Distribution;

    /// returns the distribution object according to current obs and constraint
    fn dist<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist;
    /// return the state value according to given obs
    fn value(&self, obs: Self::Obs) -> Tensor<1>;

    /// returns the distribution and state value simultaneousely. Efficient when the encoder is shared.
    fn forward<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, Tensor<1>);

    /// sample an action from actor
    fn action<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> <Self::Dist as Distribution>::Sample {
        self.valid().dist(obs, constraint).sample()
    }

    /// checks whether this actor-critic shares encoder
    fn encoder_type(&self) -> EncoderType;
}

#[derive(Debug, Clone, PartialEq)]
/// struct for determining if the actor-critic method uses separated encoder or shared encoder
pub enum EncoderType {
    /// The actor and critic shares encoder
    Shared,
    /// The actor and critic uses separated encoder
    Separated
}