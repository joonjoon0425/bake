use std::marker::PhantomData;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use std::fmt::Debug;
use crate::core::constraint::BatchedDiscreteConstraint;
use crate::core::data::BatchOps;
use crate::deep::loss::Loss;
use crate::deep::network::Network;
use crate::deep::contract::DiscreteActionValue;
use crate::deep::network::sequential::ForwardModule;

/// interpret `ForwardModule` as `qnet`
#[derive(Debug, Module)]
pub struct QNetWrapper<M: ForwardModule<In, Tensor<2>>, In: Module> {
    /// inner module
    pub m: M,
    ///
    #[module(skip)]
    pub _p: PhantomData<In>,
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module> QNetWrapper<M, In> {
    /// create new `QNetWrapper` with given module `m`
    pub fn new(m: M) -> Self { Self { m, _p: PhantomData } }

    /// update the inner module with given loss, learning rate and optimizer
    pub fn update<L: Loss>(self, loss: L, lr: f64, opt: &mut ModuleOptimizer) -> Self {
        let loss = loss.loss();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self);
        opt.step(lr, self, grads)
    }
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> Network for QNetWrapper<M, In> {
    type Obs = In;
}

impl<M: ForwardModule<In, Tensor<2>>, In: Module + BatchOps> DiscreteActionValue for QNetWrapper<M, In> {
    fn action_values<C: BatchedDiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> Tensor<2> {
        constraint.apply(self.m.forward(obs), -1e9)
    }
}