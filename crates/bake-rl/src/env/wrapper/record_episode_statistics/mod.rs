//! Episode statistics recording wrapper

mod batched;
mod single;

pub use batched::BatchedRecordEpisodeStatistics;
pub use single::RecordEpisodeStatistics;