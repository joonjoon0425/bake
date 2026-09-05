//! greedy policy implementation
use burn::prelude::*;

use crate::{constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction, explore::Exploration};

/// A greedy policy implementation
pub struct Greedy;

impl Exploration for Greedy {
    fn sample<Q: DiscreteQFunction>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = qfunc.valid().forward(obs, constraint);
        qvalues.argmax(1).squeeze_dim(1)
    }
}