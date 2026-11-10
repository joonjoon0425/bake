//! environment traits

pub mod space;

mod single;
mod batched;

pub use single::{Environment, EnvironmentConfig};
pub use batched::{BatchedEnvironment, BatchedEnvironmentConfig};