//! The algorithms of deep rl
//! 
//! 
pub mod loss_enum;

pub mod dqn;
pub use dqn::{Dqn, DqnLoss};

pub mod double_dqn;
pub use double_dqn::{DoubleDqn, DoubleDqnLoss};

pub mod reinforce;
pub use reinforce::{Reinforce, ReinforceLoss};