//! # Tabular algorithm implementations

pub mod qlearning;
pub use qlearning::*;

pub mod sarsa;
pub use sarsa::*;

pub mod expected_sarsa;
pub use expected_sarsa::*;

pub mod nstepsarsa;
pub use nstepsarsa::*;

pub mod nstepqlearning;
pub use nstepqlearning::*;