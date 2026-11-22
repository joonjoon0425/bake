use std::marker::PhantomData;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use std::fmt::Debug;
use crate::core::data::BatchOps;
use crate::deep::distribution::{Categorical, PossibleConstraint};
use crate::deep::loss::Loss;
use crate::deep::network::Network;
use crate::deep::contract::Policy;
use crate::deep::network::sequential::ForwardModule;

/// interpret `ForwardModule` as `policy` with `Categorical` distribution
#[derive(Debug, Module)]
pub struct CategoricalPolicyWrapper<M: ForwardModule<In, Tensor<2>>, In: Module> {
    /// inner module
    pub m: M,
    ///
    #[module(skip)]
    pub _p: PhantomData<In>,
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module> CategoricalPolicyWrapper<M, In> {
    /// create new `PolicyWrapper` with given module `m`
    pub fn new(m: M) -> Self { Self { m, _p: PhantomData } }
    
    /// update the inner module with given loss, learning rate and optimizer
    pub fn update<L: Loss>(self, loss: L, lr: f64, opt: &mut ModuleOptimizer) -> Self {
        let loss = loss.loss();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self);
        opt.step(lr, self, grads)
    }
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> Network for CategoricalPolicyWrapper<M, In> {
    type Obs = In;
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> Policy for CategoricalPolicyWrapper<M, In> {
    type Dist = Categorical;

    fn dist<C: PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist {
        let params = self.m.forward(obs);
        C::create_distribution(params, constraint)    
    }
}
