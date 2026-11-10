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
    
    
    pub use crate::core::constraint::BatchedDiscreteConstraint;    
    pub use crate::core::env::{Environment, BatchedEnvironment, EnvironmentConfig, BatchedEnvironmentConfig};
    pub use crate::core::data::{Shaped, BatchedTransition, key};

    pub use crate::env::BatchedTape;
    pub use crate::exploration::{BatchedDiscreteActionExploration, BatchedEpsilonGreedy, BatchedBoltzmann, BatchedGreedy};
    
    pub use bake_rl_macros::{actor_critic, policy, qnet};
}
