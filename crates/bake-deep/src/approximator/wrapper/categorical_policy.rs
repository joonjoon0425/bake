//! A PolicyNet wrapper for Categorical Distributions
//! 

use std::marker::PhantomData;
use burn::{Tensor, module::Module};
use crate::{approximator::Policy, constraint::DiscreteConstraint, distribution::Categorical, exploration::NoiseReset, network::PolicyNet};
/// A PolicyNet wrapper for Categorical Distributions
#[derive(Module, Debug)]
pub struct CategoricalPolicy<T: PolicyNet<Params = Tensor<2>>, C: DiscreteConstraint> {
    net: T,
    #[module(skip)]
    c: PhantomData<C>,
}

impl<T: PolicyNet<Params = Tensor<2>>, C: DiscreteConstraint> CategoricalPolicy<T, C> {
    /// create a new Categorical Policy
    pub fn new(net: T) -> Self { Self {net, c: PhantomData} }
}

impl<T: PolicyNet<Params = Tensor<2>> + NoiseReset, C: DiscreteConstraint> NoiseReset for CategoricalPolicy<T, C> { fn reset_noise(&mut self) { self.net.reset_noise(); } }

impl<T: PolicyNet<Params = Tensor<2>>, C: DiscreteConstraint> Policy for CategoricalPolicy<T, C> {
    type Obs = T::Obs;
    type Dist = Categorical;
    type Constraint = C;

    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        let logits = self.net.forward(obs);
        Categorical::new(logits, constraint)
    }
}