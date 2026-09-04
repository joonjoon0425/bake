//! An ActorCriticNet wrapper for Categorical Distribution
//! 
use std::marker::PhantomData;
use burn::{Tensor, module::Module};
use crate::{contract::actor_critic::ActorCritic, distribution::{Distribution, PossibleConstraint}, net::ActorCriticNet};

/// An ActorCriticNet wrapper
#[derive(Module, Debug)]
pub struct ActorCriticWrapper<T: ActorCriticNet<Params = Dist::Params>, Dist: Distribution> {
    net: T,
    #[module(skip)]
    _p: PhantomData<Dist>,
}

impl<T: ActorCriticNet<Params = Dist::Params>, Dist: Distribution> ActorCriticWrapper<T, Dist> {
    /// create a new actor critic
    pub fn new(net: T) -> Self {
        Self { net, _p: PhantomData }
    }
}

impl<T: ActorCriticNet<Params = Dist::Params>, Dist: Distribution> ActorCritic for ActorCriticWrapper<T, Dist> {
    type Obs = T::Obs;
    type Dist = Dist;

    fn dist<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist {
        let params = self.net.params(obs);
        C::create_distribution(params, constraint)
    }

    fn value(&self, obs: Self::Obs) -> Tensor<1> {
        self.net.values(obs)
    }

    fn forward<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, Tensor<1>) {
        let (logits, values) = self.net.forward(obs);
        (C::create_distribution(logits, constraint), values)
    }

    fn encoder_type(&self) -> crate::contract::actor_critic::EncoderType { self.net.encoder_type() }
}