//! `logger` module: logger for `bake-rl`

pub mod collector;
pub mod sink;

pub use collector::LogCollector;
pub use sink::{StdoutSink, CsvSink, JsonLineSink};