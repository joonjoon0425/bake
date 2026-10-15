//! # `bake-rl`: reinforcement learning framwork in Rust.
//! `bake-rl` is a reinforcement learning framework in Rust, built on `burn`.
//! - components:
//!     - `core` module: core components of `bake-rl`
//!     - `env` module: collection of reinforcement learning environments
//!         - `gym` module: collection of Farama Foundation's gymnasium environments, binded with `PyO3`
//!     - `deep` module: collection of deep reinforcement learning algorithms
//!     - `linear` module: collection of TD-linear reinforcement learning algorithms
//!     - `tabular` module: collection of tabular reinforcement learning algorithms
#![warn(missing_docs)]
pub mod core;
pub mod env;

pub mod deep;
pub mod linear;
pub mod tabular;