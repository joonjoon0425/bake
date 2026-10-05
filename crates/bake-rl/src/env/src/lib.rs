//! The environment crate for bake-rl


#![warn(missing_docs)]
pub mod env;
pub use env::*;

pub mod collection;

#[cfg(feature = "deep")]
pub mod vectorized;