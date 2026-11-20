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
    /// dump all logs to `Sink` and clear logs
    pub fn dump_to(&mut self, sink: &mut impl LogSink) -> Result<(), String> {
        let logs = std::mem::take(&mut self.logs);
        let mut vec: Vec<_> = logs.into_iter().map(|(k, (o, v))| (k, o, v)).collect();
        vec.sort_by(|a, b| a.1.cmp(&b.1) );
        for (key, _, log) in vec {
            match log {
                LogLeaf::Scalar(log) => sink.write(key, log)?,
                LogLeaf::Tensor(log) => sink.write(key, log.into_scalar::<f32>() as f64)?,
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
    Tensor(Tensor<1>)
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

impl IntoLog for Tensor<1> { fn into_log(self) -> LogLeaf { LogLeaf::Tensor(self) } }