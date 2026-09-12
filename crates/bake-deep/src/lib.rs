//! # A framework for deep reinforcement learning
//! 
#![warn(missing_docs)]

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

pub mod logger {
    //! re-exportation of bake_common::logger
    pub use bake_common::logger::*;
}

pub mod scheduler {
    //! re-exportation of bake_common::scheduler
    pub use bake_common::scheduler::*;
}