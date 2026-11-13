//! environment traits

pub mod space;
pub mod space_transform;

mod single;
mod batched;

pub use single::{Environment, EnvironmentConfig};
pub use batched::{BatchedEnvironment, BatchedEnvironmentConfig};