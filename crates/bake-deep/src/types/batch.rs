//! A Batched Transition Implementation
use burn::Tensor;

use crate::types::{NoMask};

/// A Batched `Transition` struct
#[derive(Debug, Clone)]
pub struct Batch<Obs, Action, Mask = NoMask, Extra = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: Tensor<1>,
    pub next_obss: Obs,
    pub terminated: Tensor<1>,
    pub truncated: Tensor<1>,
    pub masks: Mask,
    pub next_masks: Mask,
    pub extras: Extra,

    pub batch_size: usize,
}