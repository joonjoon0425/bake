//! log sink

/// A sink (writer) trait for logs
pub trait LogSink {
    /// flush and clear
    fn flush(&mut self) -> Result<(), String>;
    /// write value to temporary buffer
    fn write(&mut self, key: String, value: f64) -> Result<(), String>;
}

mod stdout;
pub use stdout::StdoutSink;