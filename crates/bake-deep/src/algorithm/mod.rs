pub mod dqn;
pub use dqn::{Dqn, DqnLoss};

pub mod double_dqn;
pub use double_dqn::*;

pub mod vpg;
pub use vpg::*;

pub mod a2c;
pub use a2c::*;

pub mod ppo;
pub use ppo::*;