//! An ActorCriticNet wrapper for Categorical Distribution
//! 
use std::marker::PhantomData;
use burn::{Tensor, module::Module};
use crate::{contract::actor_critic::ActorCritic, distribution::{Distribution, PossibleConstraint}, net::{ActorCriticNet, layer::NoiseReset}};

/// An ActorCriticNet wrapper
#[derive(Module, Debug)]
pub struct ActorCriticWrapper<T: ActorCriticNet<Params = Dist::Params>, Dist: Distribution> {
    net: T,
    #[module(skip)]
    _p: PhantomData<Dist>,
}

impl<T: ActorCriticNet<Params = Dist::Params>, Dist: Distribution> ActorCriticWrapper<T, Dist> {
    /// create a new actor critic with given custom network
    /// # Warning
    /// - The given network must be on the autodiff device
    /// - The network will be translated to inner device here.
    /// - The network will be automatically moved to autodiff device when the user call the `loss` functions of algorithms
    /// - The network will be automatically moved to inner device when the user call the `update` functions of algorithms
    pub fn new(net: T) -> Self {
        Self { net: net.valid(), _p: PhantomData }
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

impl<T: ActorCriticNet<Params = Dist::Params> + NoiseReset, Dist: Distribution> NoiseReset for ActorCriticWrapper<T, Dist> {
    fn reset_noise(&mut self) {
        self.net.reset_noise();
    }
}