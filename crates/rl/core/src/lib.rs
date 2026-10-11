//! # The core library component for bake-rl

#![warn(missing_docs)]
pub mod constraint;
pub mod data;

#[cfg(feature = "deep")]
pub mod deep;