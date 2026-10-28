//! epsilon-greedy policy
//!
use crate::core::{constraint::discrete::DiscreteConstraint, data::{AttachBatchDim, BatchTensor, DetachBatchDim}};
use burn::{Tensor, tensor::{Distribution, Int}};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::deep::contract::DiscreteActionValue;
use super::Exploration;

/// An epsilon-greedy policy implementation
pub struct EpsGreedy {
    eps: f32,
    rng: SmallRng,
}

impl EpsGreedy {
    /// Create a new `EpsGreedy` policy. The epsilon is the probability of choosing action randomly
    pub fn new(seed: u64, eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the eps must be between 0.0 and 1.0. Current eps is {}", eps) }
        Self {
            eps,
            rng: SmallRng::seed_from_u64(seed)
        }
    }

    /// get the value of epsilon
    pub fn eps(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl Exploration for EpsGreedy {
    /// sample an action from given Q values.
    fn sample<Q: DiscreteActionValue>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> BatchTensor<1, Int> {
        let (qvalues, bs) = qfunc.action_values(obs, constraint.clone()).detach_batch_dim();
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);
            let (masked, bs) = constraint.apply(random.attach_batch_dim(bs), -1f32).detach_batch_dim();
            masked.argmax(1).squeeze_dim(1).attach_batch_dim(bs)
        } else {
            qvalues.argmax(1).squeeze_dim(1).attach_batch_dim(bs)
        }
    }
}