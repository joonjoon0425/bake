//! writer for buffers

pub mod ring;
pub mod linear;

/// writer trait for buffer writer
pub trait Writer {
    /// return the indices where each data are written to
    /// # Arguments
    /// - capacity: the total capacity of buffer
    /// - head: the position where the data is written to
    /// - data_len: the length of the data which will be written onto buffer
    /// # Returns
    /// - returns (next head, written indices)
    fn index(&self, capacity: usize, head: usize, data_len: usize) -> Result<(usize, Vec<usize>), String>;
}