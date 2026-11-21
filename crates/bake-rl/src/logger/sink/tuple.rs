use super::LogSink;

impl<A: LogSink, B: LogSink> LogSink for (A, B) {
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()?;
        self.1.flush()?;
        Ok(())
    }

    fn write(&mut self, key: String, value: f64) -> std::io::Result<()> {
        self.0.write(key.clone(), value)?;
        self.1.write(key, value)?;
        Ok(())
    }
}

impl<A: LogSink, B: LogSink, C: LogSink> LogSink for (A, B, C) {
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()?;
        self.1.flush()?;
        self.2.flush()?;
        Ok(())
    }

    fn write(&mut self, key: String, value: f64) -> std::io::Result<()> {
        self.0.write(key.clone(), value)?;
        self.1.write(key.clone(), value)?;
        self.2.write(key, value)?;
        Ok(())
    }
}