//! frequently used data types and traits

// pub mod batchable;
// pub use batchable::Batchable;

pub mod batch;
pub use batch::Batch;

pub mod batch_dim;
pub use batch_dim::{
    BatchTensor,
    BatchEmpty,
    BatchShape,
    WithBatchDim,
};