//! epsilon-greedy policy
//!
use crate::core::{constraint::discrete::DiscreteConstraint};
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
    fn sample<Q: DiscreteActionValue>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues= qfunc.action_values(obs, constraint.clone());
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);
            let masked = constraint.apply(random, -1f32);
            masked.argmax(1).squeeze_dim(1)
        } else {
            qvalues.argmax(1).squeeze_dim(1)
        }
    }
}