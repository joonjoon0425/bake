//! frequently used data types and traits

// pub mod batchable;
// pub use batchable::Batchable;
pub mod batch_ops;
pub mod batch;

pub use batch::{
    Batch,
    transition::Transition,
    dict::BatchDict,
    dict::key,
};
pub use batch_ops::BatchOps;