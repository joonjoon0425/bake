//! environment trait

pub mod space;
pub mod wrapper;
pub mod classic;
#[cfg(feature = "gym")]
pub mod gym;

mod batched_tape;
pub use batched_tape::BatchedTape;

mod tape;
pub use tape::Tape;

mod env;
pub use env::*;

mod batched_env;
pub use batched_env::*;