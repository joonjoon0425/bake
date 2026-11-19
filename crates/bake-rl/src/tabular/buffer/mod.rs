//! buffer implementations for tabular algorithms

pub mod episode;
pub use episode::EpisodeBuffer;

pub mod window;
pub use window::WindowBuffer;