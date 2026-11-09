//! transition (s, a, r, s', ...) types

mod single;
mod batched;

pub use single::Transition;
pub use batched::BatchedTransition;