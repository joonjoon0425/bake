//! linear writer

use super::Writer;

/// linear writer
pub struct LinearWriter;

impl Writer for LinearWriter {
    fn index(&self, capacity: usize, head: usize, data_len: usize) -> Result<(usize, Vec<usize>), String> {
        if capacity < data_len { return Err("LinearWriter::index: capacity is smaller than data_len".into()) }
        else if capacity <= head { return Err("LinearWriter::index: capacity is not bigger than head (the buffer is full)".into()) }

        let capability = capacity - head;
        if data_len <= capability {
            return Ok((head + data_len, (head..head + data_len).collect()))
        } else {
            return Err(format!("LinearWriter::index: data length too long for buffer; head: {head}, capacity: {capacity}, data_len: {data_len}"))
        }
    }
}