use std::marker::PhantomData;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use std::fmt::Debug;
use crate::core::data::BatchOps;
use crate::deep::distribution::{Categorical, PossibleConstraint};
use crate::deep::loss::ActorCriticLoss;
use crate::deep::network::Network;
use crate::deep::contract::{ActorCritic, Policy, StateValue};
use crate::deep::network::sequential::ForwardModule;

/// interpret `ForwardModule` as `actor_critic` with `Categorical` distribution
#[derive(Debug, Module)]
pub struct CategoricalActorCriticWrapper<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module> {
    /// inner policy module
    pub p: M1,
    /// inner critic moudle
    pub c: M2,
    ///
    #[module(skip)]
    pub _p: PhantomData<In>,
}

impl<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module> CategoricalActorCriticWrapper<M1, M2, In> {
    /// create new `CategoricalActorCriticWrapper` with given `policy` and `critic`
    pub fn new(policy: M1, critic: M2) -> Self { Self { p: policy, c: critic, _p: PhantomData } }

    /// update the inner module with given loss, learning rate and optimizer
    pub fn update<L: ActorCriticLoss>(self, loss: L, lr_a: f64, lr_c: f64, opt_a: &mut ModuleOptimizer, opt_c: &mut ModuleOptimizer) -> Self {
        let loss_p = loss.policy_loss_with_entropy();
        let grads = loss_p.backward();
        let grads = GradientsParams::from_grads(grads, &self);
        let net = opt_a.step(lr_a, self, grads);
        let loss_c = loss.value_loss();
        let grads = loss_c.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        opt_c.step(lr_c, net, grads)
    }
}

impl<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> Network for CategoricalActorCriticWrapper<M1, M2, In> {
    type Obs = In;
}

impl<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> Policy for CategoricalActorCriticWrapper<M1, M2, In> {
    type Dist = Categorical;

    fn dist<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist {
        let params = self.p.forward(obs);
        C::create_distribution(params, constraint)
    }
}

impl<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> StateValue for CategoricalActorCriticWrapper<M1, M2, In> {
    fn state_value(&self, obs: Self::Obs) -> Tensor<1> {
        self.c.forward(obs).squeeze_dim(1)
    }
}

impl<M1: ForwardModule<In, Tensor<2>>, M2: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> ActorCritic for CategoricalActorCriticWrapper<M1, M2, In> {
    fn dist_and_state_value<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, Tensor<1>) {
        let dist = self.dist(obs.clone(), constraint);
        let state_value = self.state_value(obs);

        (dist, state_value)
    }
}