//! conversion between non-`WithBatchDim` and `WithBatchDim`

use crate::core::data::{BatchShape, WithBatchDim};

/// attach batch dimension; this will have blanket impl for already-`WithBatchDim` types
pub trait AttachBatchDim {
    /// the type after batch dimension has been attached
    type Attached: WithBatchDim;
    /// attach batch dimension
    /// # Panics
    /// - When given batch_shape is invalid
    /// # Return
    /// - attached type
    fn attach_batch_dim(self, batch_shape: BatchShape) -> Self::Attached;
}

/// blanket impl
impl<T: WithBatchDim> AttachBatchDim for T {
    type Attached = Self;

    fn attach_batch_dim(self, batch_shape: BatchShape) -> Self::Attached {
        self.reshape(batch_shape)
    }
}

/// detach batch dimension, if possible; Unlike `AttachBatchDim`, this do not have blanket impl
pub trait DetachBatchDim: WithBatchDim {
    /// the type after batch dimension has been detached
    type Detached;
    /// detach batch dimension
    /// # Return
    /// - detached type
    /// - detached batch diension
    fn detach_batch_dim(self) -> (Self::Detached, BatchShape);
}