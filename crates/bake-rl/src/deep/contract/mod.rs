//! contracts which the algorithms requires

pub mod actor_critic;
pub use actor_critic::ActorCritic;

pub mod discrete_action_value;
pub use discrete_action_value::DiscreteActionValue;

pub mod policy;
pub use policy::Policy;

pub mod state_value;
pub use state_value::StateValue;