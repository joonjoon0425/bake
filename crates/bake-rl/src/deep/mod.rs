//! # `deep`: collection of deep reinforcement learning algorithms
//! 

pub mod algorithms;
pub mod buffer;
pub mod contract;
pub mod distribution;
pub mod loss;
pub mod network;

/// prelude module
pub mod prelude {
    pub use crate::deep::algorithms::{
        dqn::*, ppo::*, advantage_estimator::*, double_dqn::*, a2c::*,
    };
    pub use crate::deep::contract::*;
    pub use crate::deep::loss::{LossFn, Loss, ActorCriticLoss};
    pub use crate::deep::network::Network;
    pub use crate::deep::buffer::sampler::{
        Sampler,
        EpochSampler,
        uniform::UniformSampler,
        suffle::ShuffleSampler,
    };
    pub use crate::deep::buffer::{
        replay::ReplayBuffer,
        rollout::RolloutBuffer,
    };
    pub use crate::deep::distribution::{Distribution, Categorical};
    
    pub use crate::core::action_selector::{BurnDiscreteActionSelector, EpsilonGreedy, Boltzmann, Greedy};
    pub use crate::core::constraint::BurnDiscreteConstraint;
    pub use crate::core::env::{Tape, VectorTape};
    pub use crate::core::env::{Environment, VectorEnvironment, EnvironmentConfig, VectorEnvironmentConfig};
    pub use crate::core::data::{Batch, BatchDict, BurnTransition, key};

    pub use bake_rl_macros::{actor_critic, policy, qnet};
}
