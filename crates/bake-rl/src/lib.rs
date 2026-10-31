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

/// re-exportation
extern crate self as bake_rl;
pub extern crate burn;

/// prelude module
pub mod prelude {
    pub use bake_rl::core::data::{key, BatchDict, Transition, Batch, BatchOps};
    pub use bake_rl::core::constraint::DiscreteConstraint;
    pub use bake_rl::core::env::{
        Tape, Environment, EnvironmentConfig, VectorEnvironment, VectorEnvironmentConfig,
        space::Space,
    };
    pub use bake_rl::core::distribution::Distribution;
    pub use bake_rl::core::buffer::sampler::{Sampler, SamplerConfig};
    pub use bake_rl::core::buffer::{
        replay::ReplayBufferConfig,
        rollout::RolloutBuffer,
    };
    pub use bake_rl::core::exploration::{
        Boltzmann,
        EpsGreedy,
        Greedy,
        Exploration,
    };
    pub use bake_rl::core::scheduler::{
        Scheduler,
        LinearScheduler,
    };
    pub use bake_macros::{BatchOps, actor_critic, qnet, policy};
}