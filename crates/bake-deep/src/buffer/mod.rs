//! # Buffers for Deep RL
//! - ReplayBuffer for DQN-like algorithms
//! - RolloutBuffer for Actor-critic algorithms
//! - EpisodeBuffer for Monte Carlo algorithms
//! 
//! 

pub mod replay_buffer;
pub use replay_buffer::*;

pub mod rollout_buffer;
pub use rollout_buffer::*;