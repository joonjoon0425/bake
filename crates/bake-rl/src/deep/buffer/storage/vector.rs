//! vector storage

use crate::core::data::{Batch, BatchOps};

/// vector storage
/// - used for episodic buffer
pub struct VectorStorage<T: BatchOps> {
    #[allow(dead_code)]
    storage: Vec<Batch<T>>,
}