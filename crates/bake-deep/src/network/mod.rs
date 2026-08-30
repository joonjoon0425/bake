//! A Network traits for algorithms
//! For custom networks, user should implement these traits, and wrap it with Masked versions
//! 

pub mod qnet;
pub use qnet::*;

pub mod policy_net;
pub use policy_net::*;

pub mod actorcritic_net;
pub use actorcritic_net::*;

pub mod basic;
pub use basic::*;