//! schedulers for `bake-rl`

/// scheduler trait
pub trait Scheduler {
    /// the type of value which is scheduled
    type Scheduled;
    /// get the value of given step
    fn value(&self, step: usize) -> Self::Scheduled;
}

mod linear;
pub use linear::LinearScheduler;

mod exponential;
pub use exponential::ExponentialScheduler;