//! # A reinforcement learning framework for rust
//! - This is a reinforcement learning crate of `bake` project.
//! - It consists of:
//!     - `core`: a common component
//!     - `env`: collection of rl environments
//!         - `gym`: collection of Farama Foundations's Gymnasium environment, using `PyO3`
//!     - `tabular`: collection of tabular rl algorithms
//!     - `deep`: collection of deep rl algorithms
//!     
#![warn(missing_docs)]
pub mod core;
pub mod env;
pub mod tabular;

// #[cfg(feature = "deep")]
pub mod deep;