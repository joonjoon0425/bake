//! convert between tensor constraint and raw discrete constraint

use crate::data::deep::Batchable;
use burn::prelude::*;

/// convert to batchable type
pub trait ToBatchable {
    /// batched type
    type Batched: Batchable;
    /// convert to batchable type
    fn to_batchable(self, device: &Device) -> Self::Batched;
}