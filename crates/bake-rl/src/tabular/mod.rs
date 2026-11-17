//! # tabular: collection of tabular reinforcement learning algorithms

pub mod algorithms;
pub mod buffer;
pub mod qtable;

/// prelude module
pub mod prelude {
    pub use crate::tabular::algorithms::{
        q_learning::*, sarsa::*, monte_carlo::*,
    };
    pub use crate::tabular::qtable::QTable;
    pub use crate::tabular::buffer::*;
    
    pub use crate::core::constraint::DiscreteConstraint;
    pub use crate::core::env::{Environment, EnvironmentConfig};
    pub use crate::core::data::Transition;

    pub use crate::exploration::{DiscreteActionExploration, EpsilonGreedy, Boltzmann, Greedy};
    pub use crate::env::Tape;
}
