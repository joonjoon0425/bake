//! one-hot observation wrapper

mod batched;
mod single;

pub use batched::BatchedOneHotObservation;
pub use single::OneHotObservation;