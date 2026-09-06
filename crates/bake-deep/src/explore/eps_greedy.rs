//! epsilon-greedy policy
//!
use burn::{Tensor, tensor::{Distribution, Int}};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction, explore::Exploration};

/// An epsilon-greedy policy implementation
pub struct EpsGreedy {
    eps: f32,
    rng: SmallRng,
}

impl EpsGreedy {
    /// Create a new `EpsGreedy` policy. The epsilon is the probability of choosing action randomly
    pub fn new(seed: u64, eps: f32) -> Self {
        assert!(0.0 <= eps && eps <= 1.0);
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
    fn sample<Q: DiscreteQFunction>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = qfunc.forward(obs, constraint.clone());
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);
            constraint.apply(random, -1f32).argmax(1).squeeze_dim(1)
        } else {
            qvalues.argmax(1).squeeze_dim(1)
        }
    }
}