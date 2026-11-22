//! wrappers for `Sequential` which enables the interpretation of `Sequential` as qnet, policy, actor-critic, etc...

mod qnet;
mod policy;
mod actor_critic;

pub use qnet::QNetWrapper;
pub use policy::CategoricalPolicyWrapper;
pub use actor_critic::CategoricalActorCriticWrapper;