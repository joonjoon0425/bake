//! A logits network for policy-based agents
use burn::module::{AutodiffModule, ModuleDisplay};

use crate::{distribution::Distribution,types::Batchable};
pub trait Policy : AutodiffModule + Clone + ModuleDisplay {
    type Obs: Batchable;
    type Dist: Distribution;
    type Constraint: Batchable;

    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist;

    fn action(&self, obs: Self::Obs, constraint: Self::Constraint) -> <Self::Dist as Distribution>::Action {
        self.forward(obs, constraint).sample()
    }
}