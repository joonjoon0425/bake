//! # `core`: core components of `bake-rl`
//!

/// representation struct for rust native types
#[derive(Debug, Clone, Copy)]
pub struct Native;
/// representation struct for `burn` types
#[derive(Debug, Clone, Copy)]
pub struct Burn;

pub mod constraint;
pub mod data;
pub mod env;
pub mod action_selector;

pub mod scheduler;