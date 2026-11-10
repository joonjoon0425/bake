//! Episode buffer for episodic algorithms
//! 
use crate::core::data::{Shaped, BatchOps};
/// Episode buffer implementation.
pub struct EpisodeBuffer<T: BatchOps> {
    batch: Vec<Shaped<T>>,
}

impl<T: BatchOps> EpisodeBuffer<T> {
    /// create a new `RolloutBuffer`
    pub fn new() -> Self {
        Self { batch: vec![] }
    }

    /// return the current length of buffer
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// push one transition into buffer
    pub fn push(&mut self, t: Shaped<T>) {
        self.batch.push(t);
    }

    /// pop all elements of buffer
    pub fn pop(&mut self) -> Shaped<T> {
        let batch = std::mem::replace(&mut self.batch, vec![]);
        Shaped::stack(batch)
    }
}