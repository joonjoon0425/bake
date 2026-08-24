//! A logits network for policy-based agents
use burn::module::{AutodiffModule, ModuleDisplay};
use crate::{distribution::Distribution,types::Batchable};

/// a parametrized policy for policy gradient methods
pub trait Policy : AutodiffModule + Clone + ModuleDisplay {
    /// the observation of environment
    type Obs: Batchable;
    /// the distribution which policy produces
    type Dist: Distribution;
    /// the constraint associated with observation
    type Constraint: Batchable;

    /// get the distribution of actions
    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist;

    /// get the current action from given observation and constraint
    fn action(&self, obs: Self::Obs, constraint: Self::Constraint) -> <Self::Dist as Distribution>::Action {
        self.forward(obs, constraint).sample()
    }
}