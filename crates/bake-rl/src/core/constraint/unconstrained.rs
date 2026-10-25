use crate::core::data::{BatchEmpty, BatchShape, WithBatchDim, batch_dim::convert::{AttachBatchDim, DetachBatchDim}};

/// Unconstrained constraint
#[derive(Debug, Clone, Copy)]
pub struct Unconstrained;

impl AttachBatchDim for Unconstrained {
    type Attached = BatchEmpty<Unconstrained>;
    
    fn attach_batch_dim(self, batch_shape: BatchShape) -> Self::Attached {
        BatchEmpty::with_batch_dim(batch_shape)
    }
}

impl DetachBatchDim for BatchEmpty<Unconstrained> {
    type Detached = Unconstrained;

    fn detach_batch_dim(self) -> (Self::Detached, BatchShape) {
        (Unconstrained, self.batch_shape())
    }
}