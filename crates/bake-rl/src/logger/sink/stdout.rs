use super::LogSink;

/// stdout sink
pub struct StdoutSink {
    heading: String,
    buf: String,
}

impl StdoutSink { 
    /// create new `StdoutSink`
    pub fn new(title: impl Into<String>) -> Self {
        let title = title.into();
        let heading = format!("{}\n\n\x1b[1m{:^80}\x1b[0m\n\n", "#".repeat(80), title);
        Self { buf: heading.clone(), heading, }
    }
}

impl LogSink for StdoutSink {
    fn write(&mut self, key: String, value: f64) -> Result<(), String> {
        let val = if value.is_finite() && value.fract() == 0.0 { format!("{:>40}: {:.0}", key, value) } else { format!("{:>40}: {:.8}", key, value) };
        self.buf += &format!("{}\n", val);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        println!("{}", self.buf);
        println!("{}", "#".repeat(80));
        self.buf = self.heading.clone();
        Ok(())
    }
}