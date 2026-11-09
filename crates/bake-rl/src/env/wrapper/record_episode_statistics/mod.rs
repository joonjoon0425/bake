//! Episode statistics recording wrapper

mod batched;
mod single;

/// episode statistics
/// - id: ID
/// - r: accumulated rewards
/// - s: accumulated steps
#[derive(Debug, Clone)]
pub struct EpisodeStatistics {
    /// environment id
    pub id: usize,
    /// accumulated rewards
    pub r: f32,
    /// accumulated steps
    pub s: i64,
}


pub use batched::BatchedRecordEpisodeStatistics;
pub use single::RecordEpisodeStatistics;