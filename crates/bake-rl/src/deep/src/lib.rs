//! # A framework for deep reinforcement learning
//!
//!
#![warn(missing_docs)]

extern crate self as bake_deep;
pub extern crate burn;

pub mod algorithm;
pub mod buffer;
pub mod contract;
pub mod distribution;
pub mod explore;
pub mod loss;
pub mod net;

pub mod logger {
    //! re-exportation of bake_common::logger
    pub use bake_common::logger::*;
}

pub mod scheduler {
    //! re-exportation of bake_common::scheduler
    pub use bake_common::scheduler::*;
}

pub mod macros {
    //! re-exportation of bake_macros
    pub use bake_macros::*;
}

pub mod data {
    //! re-exportation of bake_rl_core::deep::data
    pub use bake_rl_core::deep::data::*;
}

pub mod constraint {
    //! re-exportation of bake_rl_core::deep::constraint
    pub use bake_rl_core::deep::constraint::*;
}