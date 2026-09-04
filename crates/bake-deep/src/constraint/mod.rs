//! A constraint trait and types for deep rl
//! 
//! 
#[derive(Debug, Clone, Copy)]
/// A struct which indicates that all actions are possible.<br>
/// Do not mix it in environment; An environment with `Unconstrained` must always return `Unconstrained`.
pub struct Unconstrained;

pub mod discrete_constraint;