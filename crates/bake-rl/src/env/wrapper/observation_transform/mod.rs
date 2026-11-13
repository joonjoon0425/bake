//! wrapper for observation transformation

mod single;
mod batched;

pub use single::ObservationTransform;
pub use batched::BatchedObservationTransform;