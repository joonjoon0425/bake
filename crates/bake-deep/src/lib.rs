//! # A framework for deep reinforcement learning
//!
//!
#![warn(missing_docs)]

extern crate self as bake_deep;
pub extern crate burn;

pub mod algorithm;
pub mod buffer;
pub mod constraint;
pub mod contract;
pub mod data;
pub mod distribution;
pub mod env;
pub mod explore;
pub mod loss;
pub mod net;
pub mod wrapper;

pub mod experimental;

pub mod logger {
    //! re-exportation of bake_common::logger
    pub use bake_common::logger::*;
}

pub mod scheduler {
    //! re-exportation of bake_common::scheduler
    pub use bake_common::scheduler::*;
}