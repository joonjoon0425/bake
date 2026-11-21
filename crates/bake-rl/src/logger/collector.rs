//! log collector
use super::sink::LogSink;
use std::collections::BTreeMap;
use burn::prelude::*;

/// log collector
/// - collects scalar logs and tensor logs
#[derive(Debug, Clone)]
pub struct LogCollector {
    logs: BTreeMap<String, (i32, LogLeaf)>,
}

impl LogCollector {
    /// create new `LogCollector`
    pub fn new() -> Self { Self { logs: BTreeMap::new() } }
    /// record log
    /// - returns the previously saved value if it existed
    pub fn record(&mut self, order: i32, key: impl Into<String>, value: impl IntoLog) -> Option<LogLeaf> {
        let log = value.into_log();
        let ret = self.logs.insert(key.into(), (order, log));
        ret.map_or(None, |a| Some(a.1))
    }
    /// record log, reducing to mean
    /// # Warning
    /// - returns the previously saved value, if it was not recorded with `record_mean` before
    /// - the given order is ignored when the value is not recorded for the first time
    pub fn record_mean(&mut self, order: i32, key: impl Into<String>, value: impl IntoLog) -> Option<LogLeaf> {
        let log = value.into_log();
        let key = key.into();
        let prev = self.logs.get_mut(&key);
        let ret = match prev {
            None => self.logs.insert(key, (order, log)),
            Some((_, prev)) => {
                match (log, prev) {
                    (LogLeaf::Scalar(log), LogLeaf::Scalar(_)) => {
                        let log = LogLeaf::ScalarMean { value: log, count: 1 };
                        self.logs.insert(key, (order, log))
                    },
                    (LogLeaf::Scalar(log), LogLeaf::ScalarMean { value, count }) => {
                        *value += log; *count += 1;
                        None
                    },
                    (LogLeaf::Scalar(log), LogLeaf::Tensor(_)) => {
                        let log = LogLeaf::ScalarMean { value: log, count: 1 };
                        self.logs.insert(key, (order, log))
                    },
                    (LogLeaf::Scalar(log), LogLeaf::TensorMean { value, count }) => {
                        let log = Tensor::from_floats([log as f32], &value.device());
                        value.inplace(|v| v + log); *count += 1;
                        None
                    },
                    (LogLeaf::Tensor(log), LogLeaf::Scalar(_)) => {
                        let log = LogLeaf::TensorMean { value: log.inner(), count: 1 };
                        self.logs.insert(key, (order, log))
                    },
                    (LogLeaf::Tensor(log), LogLeaf::ScalarMean { value, count }) => {
                        let log = log.into_scalar::<f32>() as f64;
                        *value += log; *count += 1;
                        None
                    },
                    (LogLeaf::Tensor(log), LogLeaf::Tensor(_)) => {
                        let log = LogLeaf::TensorMean { value: log.inner(), count: 1 };
                        self.logs.insert(key, (order, log))
                    },
                    (LogLeaf::Tensor(log), LogLeaf::TensorMean { value, count }) => {
                        value.inplace(|v| v + log.inner()); *count += 1;
                        None
                    },
                    _ => panic!("Something is wrong in implementation. `impl IntoLog` should only produce the non-aggregate values")
                }
            }
        };
        ret.map_or(None, |a| Some(a.1))
    }
    /// dump all logs to `Sink` and clear logs
    /// # Warning
    /// - some `LogSink`s such as `CsvSink` follows the order of header instead of given orders when the value was recorded
    pub fn dump_to(&mut self, step: usize, sink: &mut impl LogSink) -> std::io::Result<()> {
        let logs = std::mem::take(&mut self.logs);
        let mut vec: Vec<_> = logs.into_iter().map(|(k, (o, v))| (k, o, v)).collect();
        vec.sort_by(|a, b| a.1.cmp(&b.1) );

        sink.write("step".into(), step as f64)?;

        for (key, _, log) in vec {
            match log {
                LogLeaf::Scalar(log) => sink.write(key, log)?,
                LogLeaf::Tensor(log) => sink.write(key, log.into_scalar::<f32>() as f64)?,
                LogLeaf::ScalarMean { value, count } => sink.write(key, value / count as f64)?,
                LogLeaf::TensorMean { value, count } => sink.write(key, value.into_scalar::<f32>() as f64 / count as f64)?,
            }
        }
        sink.flush()?;
        Ok(())
    }
}

/// simple log enum which `LogCollector` uses
#[derive(Debug, Clone)]
pub enum LogLeaf {
    /// scalar log
    Scalar(f64),
    /// tensor log
    Tensor(Tensor<1>),
    /// scalar mean log
    ScalarMean {
        /// sum
        value: f64,
        /// number of values
        count: u64
    },
    /// tensor mean log
    TensorMean {
        /// sum
        value: Tensor<1>,
        /// number of values
        count: u64
    }
}

/// scalar type logs
pub trait IntoLog {
    /// convert to `LogLeaf` for logging
    fn into_log(self) -> LogLeaf; 
}

macro_rules! impl_scalar_log {
    ($ty:ty) => {
       impl IntoLog for $ty { fn into_log(self) -> LogLeaf { LogLeaf::Scalar(self as f64) } } 
    };
    ($($ty:ty $(,)?)*) => {
        $(
            impl_scalar_log!($ty);
        )*
    }
}

impl_scalar_log!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

impl IntoLog for Tensor<1> { fn into_log(self) -> LogLeaf { LogLeaf::Tensor(self.inner()) } }
