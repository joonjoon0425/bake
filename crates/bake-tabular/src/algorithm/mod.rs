//! The algorithms of tabular rl
//! 
pub mod q_learning;
pub use q_learning::QLearning;

pub mod sarsa;
pub use sarsa::Sarsa;

pub mod nstep_estimator;
pub use nstep_estimator::NStepEstimator;

pub mod nstep_sarsa;
pub use nstep_sarsa::NStepSarsa;