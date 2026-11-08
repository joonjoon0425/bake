//! round robin writer

use super::Writer;

/// round robin writer
pub struct RingWriter;

impl Writer for RingWriter {
    fn index(&self, capacity: usize, head: usize, data_len: usize) -> Result<usize, String> {
        if capacity < data_len { return Err("RingWriter::index: capacity is smaller than data_len".into()) }
        else if capacity <= head { return Err("RingWriter::index: capacity is not bigger than head".into()) }

        Ok((head + data_len) % capacity)
    }
}