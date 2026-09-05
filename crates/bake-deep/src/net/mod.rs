//! Traits for user-defined, custom networks
//! 

pub mod qnet;
pub use qnet::{DiscreteQNet, DiscreteDuelingQNet};

pub mod policy_net;
pub use policy_net::{PolicyNet};

pub mod actor_critic_net;
pub use actor_critic_net::{ActorCriticNet};

pub mod layer;
pub mod basic;