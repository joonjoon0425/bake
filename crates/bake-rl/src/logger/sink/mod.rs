//! log sink

/// A sink (writer) trait for logs
pub trait LogSink {
    /// flush and clear
    fn flush(&mut self) -> std::io::Result<()>;
    /// write value to temporary buffer
    fn write(&mut self, key: String, value: f64) -> std::io::Result<()>;
}

mod stdout;
pub use stdout::StdoutSink;

mod csv;
pub use csv::CsvSink;

mod tuple;