use std::marker::PhantomData;
use burn::tensor::backend::Backend;
use crate::traits::Batchable;

pub struct ReplayBuffer<B: Backend, Obs: Batchable<B>, Action: Batchable<B>> {
    backend: PhantomData<B>
}