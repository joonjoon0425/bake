//! Collection of environments
//! 
pub mod cliff_walking;
pub use cliff_walking::{CliffWalking, CliffWalkingConfig};

pub mod masked_cliff_walking;
pub use masked_cliff_walking::{MaskedCliffWalking, MaskedCliffWalkingConfig};

pub mod cartpole;
pub use cartpole::{CartPole, CartPoleConfig};