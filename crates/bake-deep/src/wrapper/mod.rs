//! A wrapper for use-defined custom networks
//! 
pub mod actor_critic_wrapper;
pub use actor_critic_wrapper::ActorCriticWrapper;

pub mod policy_wrapper;
pub use policy_wrapper::PolicyWrapper;

pub mod qnet_wrapper;
pub use qnet_wrapper::DiscreteQNetWrapper;