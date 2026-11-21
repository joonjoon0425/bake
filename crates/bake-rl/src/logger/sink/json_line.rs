use std::io::BufWriter;
use std::io::Write;
use std::fs::File;
use std::path::Path;

use serde_json::Value;

use super::LogSink;

/// json sink
#[derive(Debug)]
pub struct JsonLineSink {
    writer: BufWriter<File>,
    buf: serde_json::Map<String, Value>,
}

impl JsonLineSink {
    /// create new `JsonSink`
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self { writer, buf: serde_json::Map::new() })
    }
}

impl LogSink for JsonLineSink {
    fn write(&mut self, key: String, value: f64) -> std::io::Result<()> {
        self.buf.insert(key.to_string(), Value::from(value));
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        serde_json::to_writer(&mut self.writer, &self.buf)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.buf.clear();
        Ok(())
    }
}