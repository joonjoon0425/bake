use std::io::Write;
use std::{fs::File, path::Path};
use std::collections::BTreeMap;
use super::LogSink;

/// csv format sink
#[derive(Debug)]
pub struct CsvSink {
    file: File,
    header: Vec<String>,
    buf: BTreeMap<String, f64>,

    once: Option<String>,
}

impl CsvSink {
    /// create new `CsvSink`
    pub fn new(path: &Path, header: &[impl Into<String> + Clone]) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut header: Vec<_> = header.iter().map(|a| a.clone().into()).collect();
        header.insert(0, "step".into());
        let mut once = header.iter().fold(String::new(), |a, b| a + b + ",");
        once += "\n";

        Ok(Self {
            file,
            once: Some(once),
            buf: BTreeMap::new(),
            header,
        })
    }
}

impl LogSink for CsvSink {
    fn flush(&mut self) -> std::io::Result<()> {
        let mut buf = self.once.take().unwrap_or_default();
        for head in &self.header {
            match self.buf.remove(head) {
                Some(value) => buf += &format!("{},", value),
                None => buf += ",",
            }
        }
        buf += "\n";
        self.file.write(&buf.into_bytes())?;
        Ok(())
    }

    fn write(&mut self, key: String, value: f64) -> std::io::Result<()> {
        if self.header.contains(&key) {
            self.buf.insert(key, value);
            Ok(())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("The given key `{}` was not registered. Please contain the key `{}` when creating `CsvSink` with `CsvSink::new()`", key, key)))
        }
    }
}