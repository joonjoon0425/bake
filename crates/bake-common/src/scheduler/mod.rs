//! Schedulers
//! - Only a schedulers about f32 and f64 are implemented here

/// trait for Scheduler implementations
pub trait Scheduler {
    /// the scheduled type
    type Scheduled;
    /// give the next step of schedule
    fn step(&mut self) -> Self::Scheduled;
    /// reset the scheduler
    fn reset(&mut self);
}

pub mod linear_scheduler;
pub use linear_scheduler::LinearScheduler;