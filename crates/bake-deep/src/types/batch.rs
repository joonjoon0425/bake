//! A Batched Transition Implementation
use crate::types::Batchable;

/// A Batched `Transition` struct
#[derive(Debug, Clone)]
pub struct Batch<Obs, Action, Mask = (), Extra = ()> {
    pub obss: Obs,
    pub actions: Action,
    pub rewards: <f32 as Batchable>::Batched,
    pub next_obss: Obs,
    pub terminated: <bool as Batchable>::Batched,
    pub truncated: <bool as Batchable>::Batched,
    pub masks: Mask,
    pub next_masks: Mask,
    pub extras: Extra,

    pub batch_size: usize,
}