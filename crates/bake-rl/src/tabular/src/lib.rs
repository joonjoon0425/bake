//! # A framework for tabular reinforcement learning
//! 
#![warn(missing_docs)]

pub mod algorithm;
pub mod buffer;
pub mod constraint;
pub mod data;
pub mod env;
pub mod explore;
pub mod qtable;

pub mod logger {
    //! re-exportation of bake_common::logger
    pub use bake_common::logger::*;
}

pub mod scheduler {
    //! re-exportation of bake_common::scheduler
    pub use bake_common::scheduler::*;
}