//! # `core`: core components of `bake-rl`
//!

/// representation struct for implementation which does not uses `burn`
#[derive(Debug, Clone, Copy)]
pub struct Native;
/// representation struct for implementations which uses `burn`
#[derive(Debug, Clone, Copy)]
pub struct Burn;

pub mod constraint;
pub mod data;
pub mod env;
pub mod action_selector;

pub mod scheduler;