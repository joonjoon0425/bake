//! # A Deep Reinforcement Learning Framework
//! Works with burn
#[warn(missing_docs)]
pub mod approximator;
pub mod algorithm;
pub mod buffer;
pub mod config;
pub mod constraint;
pub mod distribution;
pub mod env;
pub mod exploration;
pub mod network;
pub mod scheduler;
pub mod types;
pub mod utils;

pub mod prelude {
    pub use crate::network::{QNet, DuelingQNet, PolicyNet, ActorCriticNet};

    pub use crate::approximator::{QFunction, Policy, ActorCritic};

    pub use crate::exploration::{Exploration, NoiseReset};
    pub use crate::distribution::Distribution;
    pub use crate::constraint::DiscreteConstraint;
    pub use crate::types::{Batchable, Recordable};
    pub use crate::env::Env;

    pub use crate::types::{Batch, Tape, Logger, ValueLoss};
    pub use crate::constraint::{DiscreteMask, Unconstrained};

    pub use bake_macros::Batchable;
}