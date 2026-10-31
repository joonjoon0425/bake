//! predefined dictionary struct keys

use std::marker::PhantomData;
use std::borrow::Cow;
use burn::prelude::*;
use crate::core::data::batch::dict::Key;

impl<T> Key<T> {
    /// create a new const key with given name
    pub const fn new(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            _p: PhantomData,
        }
    }
}

/// key for advantage
#[allow(non_upper_case_globals)]
pub const adv: Key<Tensor<2>> = Key::new("adv");

/// key for return
#[allow(non_upper_case_globals)]
pub const ret: Key<Tensor<2>> = Key::new("ret");

/// key for log_prob
#[allow(non_upper_case_globals)]
pub const log_prob: Key<Tensor<2>> = Key::new("log_prob");