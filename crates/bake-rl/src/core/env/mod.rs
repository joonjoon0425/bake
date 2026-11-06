//! environment trait

pub mod space;
pub mod wrapper;

mod vec_tape;
pub use vec_tape::VectorTape;

mod tape;
pub use tape::Tape;

mod env;
pub use env::*;

mod vec_env;
pub use vec_env::*;