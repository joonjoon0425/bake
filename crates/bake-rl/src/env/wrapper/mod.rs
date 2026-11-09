//! Wrappers for environments

pub mod sync_env;
mod into_batched;
pub mod record_episode_statistics;
pub mod one_hot_observation;

pub use sync_env::SynchronizedBatchedEnvironment;
pub use record_episode_statistics::{RecordEpisodeStatistics, BatchedRecordEpisodeStatistics};
pub use one_hot_observation::{OneHotObservation, BatchedOneHotObservation};
