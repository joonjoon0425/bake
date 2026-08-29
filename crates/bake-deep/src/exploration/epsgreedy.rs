//! epsilon-greedy policy
//!
use burn::{Tensor, tensor::{Distribution, Int}};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::{approximator::QFunction, constraint::DiscreteConstraint, exploration::Exploration};
pub struct EpsGreedy {
    eps: f32,
    rng: StdRng,
}

impl EpsGreedy {
    /// Create a new EpsGreedy policy
    pub fn new(seed: u64, eps: f32) -> Self {
        Self {
            eps,
            rng: StdRng::seed_from_u64(seed)
        }
    }

    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl Exploration for EpsGreedy {
    /// sample an action from given Q values.
    fn sample<Q: QFunction>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = qfunc.forward(obs, constraint.clone());
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);
            constraint.apply(random, -1f32).argmax(1).squeeze_dim(1)
        } else {
            qvalues.argmax(1).squeeze_dim(1)
        }
    }

    /// give the probability of choosing the given action in given observation
    fn prob<Q: QFunction>(&self, qfunc: &Q, obs: Q::Obs, action: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<1> {
        let qvalues = qfunc.forward(obs, constraint);
        let n_valid = qvalues.clone().greater_elem(-5e8).float().sum_dim(1).squeeze_dim(1);
        let argmax = qvalues.argmax(1).squeeze_dim(1);
        action.equal(argmax).float() * (1.0 - self.eps) + self.eps / n_valid
    }
}