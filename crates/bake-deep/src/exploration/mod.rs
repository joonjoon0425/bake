//! # Exploration strategies for Deep RL
//! 
use burn::{Tensor, tensor::Int};
use crate::{approximator::QFunction, constraint::DiscreteConstraint};

pub trait Exploration {
    fn sample<QFunc: QFunction>(&mut self, qfunc: &QFunc, obs: QFunc::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int>;
    fn prob<QFunc: QFunction>(&self, qfunc: &QFunc, obs: QFunc::Obs, action: Tensor<1, Int>, constraint: impl DiscreteConstraint) -> Tensor<1>;
}
pub mod greedy;
pub use greedy::*;

pub mod epsgreedy;
pub use epsgreedy::*;

pub mod boltzmann;
pub use boltzmann::*;

pub mod noisylinear;
pub use noisylinear::*;