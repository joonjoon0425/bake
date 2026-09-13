//! The traits which algorithms requires
//! 

pub mod actor_critic;
pub use actor_critic::ActorCritic;

pub mod qfunction;
pub use qfunction::{DiscreteQFunction};

pub mod policy;
pub use policy::{Policy};