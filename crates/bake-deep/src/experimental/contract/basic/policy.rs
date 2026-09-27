//! A trait for parametrized policy
use crate::distribution::{Distribution, PossibleConstraint};
use crate::experimental::contract::basic::network::Network;
/// a parametrized policy
pub trait Policy: Network {
    /// the distribution which policy produces
    type Dist: Distribution;

    // /// get the original output of network
    // /// - the original network produces the Dist::Params
    // fn params(&self, obs: Self::Obs) -> <Self::Dist as Distribution>::Params;

    /// get the distribution
    fn dist<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist;

    /// get the current action from given observation and constraint
    fn action<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> <Self::Dist as Distribution>::Sample {
        self.dist(obs, constraint).sample()
    }
}