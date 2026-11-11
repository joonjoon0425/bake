//! space traits for Gymnasium environments

mod single;
pub use single::GymSpace;

mod batched;
pub use batched::BatchedGymSpace;