//! typed dictionary struct keys

use std::marker::PhantomData;
use std::borrow::Cow;
use burn::prelude::*;

/// typed key struct
#[derive(Debug, Clone)]
pub struct Key<T> {
    pub(crate) name: Cow<'static, str>,
    _p: PhantomData<T>
}

impl<T> Key<T> {
    /// create a new const key with given name
    pub const fn const_key(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            _p: PhantomData,
        }
    }

    /// create a new key with given name
    pub fn dyn_key(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            _p: PhantomData,
        }
    }

    /// return the name of key
    pub fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }
}

/// key for advantage
#[allow(non_upper_case_globals)]
pub const adv: Key<Tensor<1>> = Key::const_key("adv");

/// key for return
#[allow(non_upper_case_globals)]
pub const ret: Key<Tensor<1>> = Key::const_key("ret");

/// key for log_prob
#[allow(non_upper_case_globals)]
pub const log_prob: Key<Tensor<1>> = Key::const_key("log_prob");