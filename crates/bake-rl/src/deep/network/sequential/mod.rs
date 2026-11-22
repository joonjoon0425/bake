//! An experimental sequential network container implementation with tuple
use std::marker::PhantomData;
use burn::module::{Module, ModuleDisplay};
use std::fmt::Debug;
use crate::core::data::BatchOps;
use crate::deep::network::Network;

/// `burn::Module` with forward
pub trait ForwardModule<In, Out> : Module + ModuleDisplay {
    /// forward
    fn forward(&self, input: In) -> Out;
}

/// sequential module container
#[derive(Debug, Module)]
pub struct Sequential<A: ForwardModule<Input, Middle>, B: ForwardModule<Middle, Output>, Input: Debug, Middle: Debug, Output: Debug> {
    /// head of sequential
    pub head: A,
    /// tail of sequential
    pub tail: B,
    ///
    #[module(skip)]
    pub _p: PhantomData<fn()->(Input, Middle, Output)>,
}

impl<A: ForwardModule<Input, Middle>, B: ForwardModule<Middle, Output>, Input: Debug, Middle: Debug, Output: Debug> Sequential<A, B, Input, Middle, Output> {
    /// create new `Sequential`
    pub fn new(head: A, tail: B) -> Self {
        Self {
            head, tail, _p: PhantomData
        }
    }
}

impl<A: ForwardModule<Input, Middle>, B: ForwardModule<Middle, Output>, Input: Debug, Middle: Debug, Output: Debug> ForwardModule<Input, Output> for Sequential<A, B, Input, Middle, Output> {
    fn forward(&self, input: Input) -> Output {
        let m = self.head.forward(input);
        self.tail.forward(m)
    }
}

impl<A: ForwardModule<Input, Middle>, B: ForwardModule<Middle, Output>, Input: Debug + BatchOps, Middle: Debug, Output: Debug> Network for Sequential<A, B, Input, Middle, Output> {
    type Obs = Input;
}

/// creates `Sequential`
#[macro_export]
macro_rules! seq {
    ($m1:expr, $m2:expr $(,)?) => {
        ::bake_rl::deep::network::sequential::Sequential::new($m1, $m2)
    };
    ($m1:expr, $($rest:expr),+ $(,)?) => {
        ::bake_rl::deep::network::sequential::Sequential::new($m1, seq!($($rest),+))
    };
}

mod r#impl;
pub mod wrapper;