//! A trait for parametrized policy for policy gradient methods
use burn::module::{AutodiffModule, ModuleDisplay};

use crate::data::batchable::Batchable;
use crate::distribution::{Distribution, PossibleConstraint};
/// a parametrized policy for policy gradient methods
pub trait Policy: AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;
    /// the distribution which policy produces
    type Dist: Distribution;

    /// get the distribution of actions
    fn forward<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist;

    /// get the current action from given observation and constraint
    fn action<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> <Self::Dist as Distribution>::Sample {
        self.valid().forward(obs, constraint).sample()
    }
}