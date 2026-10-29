//! # `deep`: collection of deep reinforcement learning algorithms
//! 

pub mod algorithms;
pub mod contract;
pub mod loss;
pub mod network;

/// prelude module
pub mod prelude {
    pub use crate::deep::algorithms::{
        dqn::*,
    };
    pub use crate::deep::contract::*;
    pub use crate::deep::loss::{LossFn, Loss, ActorCriticLoss};
    pub use crate::deep::network::Network;
}