//! frequently used data types and traits

// pub mod batchable;
// pub use batchable::Batchable;
pub mod batch_ops;
pub mod dict;
pub mod transition;
pub mod shaped;

pub use shaped::{
    Batch,
    transition::{Transition, NativeTransition, BurnTransition},
    dict::BatchDict,
    dict::key,
};
pub use batch_ops::BatchOps;
