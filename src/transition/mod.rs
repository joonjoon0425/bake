use std::marker::PhantomData;

use burn::{Tensor, tensor::backend::Backend};

use crate::traits::Batchable;

pub struct Transition<B: Backend, Obs: Batchable<B> + Clone, Action: Batchable<B> + Clone, Extra: Batchable<B> + Clone = ()> {
    pub observation: Obs,
    pub action: Action,
    pub reward: f32,
    pub next_observation: Obs,

    pub terminated: bool,
    pub truncated: bool,
    pub extra: Extra,

    pub _backend: PhantomData<B>
}

pub struct BatchedTransition<B: Backend, Obs, Action, Extra = ()> {
    pub observations: Obs,
    pub actions: Action,
    pub rewards: Tensor<B, 1>,
    pub next_observations: Obs,

    pub terminated: Tensor<B, 1>,
    pub truncated: Tensor<B, 1>,
    pub extras: Extra,

    pub batch_size: usize,

    pub _backend: PhantomData<B>
}