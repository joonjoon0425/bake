//! round robin writer

use super::Writer;

/// round robin writer
pub struct RingWriter;

impl Writer for RingWriter {
    fn index(&self, capacity: usize, head: usize, data_len: usize) -> Result<(usize, Vec<usize>), String> {
        if capacity < data_len { return Err("RingWriter::index: capacity is smaller than data_len".into()) }
        else if capacity <= head { return Err("RingWriter::index: capacity is not bigger than head".into()) }

        let capability = capacity - head;
        if data_len <= capability {
            return Ok(((head + data_len) % capacity, (head..head + data_len).collect()))
        } else {
            return Ok((data_len - capability, (head..capacity).chain(0..data_len - capability).collect()))
        }
    }
}