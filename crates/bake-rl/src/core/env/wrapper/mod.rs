//! Wrappers for environments

pub mod vec;

pub mod sync_env;
mod into_burn;

pub mod record_episode_statistics;
pub use record_episode_statistics::RecordEpisodeStatistics;

pub use sync_env::SynchronizedVectorEnvironment;