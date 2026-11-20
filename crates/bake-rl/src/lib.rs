//! # `bake-rl`: reinforcement learning framework in Rust.
//! `bake-rl` is a reinforcement learning framework in Rust, built on `burn`.
//! - components:
//!     - `core` module: core components of `bake-rl`
//!     - `env` module: collection of reinforcement learning environments and wrappers
//!         - `gym` module: collection of Farama Foundation's gymnasium environments, binded with `PyO3`
//!     - `exploration` module: collection of exploration strategies
//!     - `logger` module: logger for `bake-rl`
//!     - `deep` module: collection of deep reinforcement learning algorithms
//!     - `linear` module: collection of TD-linear reinforcement learning algorithms
//!     - `tabular` module: collection of tabular reinforcement learning algorithms
#![warn(missing_docs)]
pub mod core;
pub mod env;
pub mod exploration;

pub mod deep;
pub mod linear;
pub mod tabular;

pub mod logger;

/// re-exportation
extern crate self as bake_rl;
pub extern crate bake_rl_macros;
pub extern crate burn;
pub extern crate rand;