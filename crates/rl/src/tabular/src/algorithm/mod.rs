//! The algorithms of tabular rl
//! 
pub mod qlearning;
pub use qlearning::QLearning;

pub mod sarsa;
pub use sarsa::Sarsa;

pub mod nstep_estimator;

pub mod nstep_sarsa;
pub use nstep_sarsa::NStepSarsa;

pub mod nstep_qlearning;
pub use nstep_qlearning::NStepQLearning;