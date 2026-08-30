//! An ActorCriticNet wrapper for Categorical Distribution
//! 
use std::marker::PhantomData;
use burn::{Tensor, module::Module};
use crate::{approximator::ActorCritic, constraint::DiscreteConstraint, distribution::Categorical, network::{ActorCriticNet, EncoderType}};
/// A PolicyNet wrapper for Categorical Distributions
#[derive(Module, Debug)]
pub struct CategoricalActorCritic<T: ActorCriticNet<Params = Tensor<2>>, C: DiscreteConstraint> {
    net: T,
    #[module(skip)]
    c: PhantomData<C>,
}

impl<T: ActorCriticNet<Params = Tensor<2>>, C: DiscreteConstraint> CategoricalActorCritic<T, C> {
    /// create a new actor critic for categorical distributions
    pub fn new(net: T) -> Self {
        Self { net, c: PhantomData }
    }
}

impl<T: ActorCriticNet<Params = Tensor<2>>, C: DiscreteConstraint> ActorCritic for CategoricalActorCritic<T, C> {
    type Obs = T::Obs;
    type Dist = Categorical;
    type Constraint = C;

    fn dist(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        let logits = self.net.params(obs);
        Categorical::new(logits, constraint)
    }

    fn value(&self, obs: Self::Obs) -> Tensor<1> {
        self.net.values(obs)
    }

    fn forward(&self, obs: Self::Obs, constraint: Self::Constraint) -> (Self::Dist, Tensor<1>) {
        let (logits, values) = self.net.forward(obs);
        (Categorical::new(logits, constraint), values)
    }

    fn encoder_type(&self) -> EncoderType { self.net.encoder_type() }
}