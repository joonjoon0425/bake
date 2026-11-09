//! frequently used data types and traits

// pub mod batchable;
// pub use batchable::Batchable;
pub mod batch_ops;
pub mod dict;
pub mod transition;
pub mod shaped;

pub use shaped::{
    Shaped,
};

pub use dict::{
    BatchedDict, DictLeaf, key,
};

pub use batch_ops::BatchOps;

pub use transition::{
    BatchedTransition, Transition
};