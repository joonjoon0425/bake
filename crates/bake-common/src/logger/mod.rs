//! Logger and Metrics
//! 

use std::collections::{HashMap, VecDeque};

/// A Moving average log struct which Logger uses internally
struct MovingAvgLog {
    pub sum: f32,
    pub values: VecDeque<f32>,
    pub window: usize,
}

/// A logger struct
pub struct MovingAvgLogger {
    moving_avg: HashMap<&'static str, MovingAvgLog>,
}

impl MovingAvgLogger {
    /// create a new `Logger`
    pub fn new() -> Self {
        Self {
            moving_avg: HashMap::default(),
        }
    }

    /// register a name and window length for moving average
    pub fn register(&mut self, name: &'static str, window: usize) {
        self.moving_avg.insert(name, MovingAvgLog { sum: 0.0, values: VecDeque::with_capacity(window), window, });
    }

    /// push a new log to slot of given name
    /// # Warning
    /// - the `window` is ignored after the first encounter of given `name`
    /// # Panic
    /// - panics when a given `name` was seen for the first time and the `window` is None
    pub fn push_single(&mut self, name: &'static str, value: f32, window: Option<usize>) {
        let moving_avg = self.moving_avg.get_mut(name);

        if moving_avg.is_some() {
            let moving_avg = moving_avg.unwrap();
            if moving_avg.values.len() < moving_avg.window {
                moving_avg.sum += value;
                moving_avg.values.push_back(value);
            } else {
                moving_avg.sum += value - moving_avg.values.front().unwrap();
                moving_avg.values.pop_front();
                moving_avg.values.push_back(value);
            }
        } else {
            let window = window.expect(&format!("The given name {} was unregistered, but the `window` size was not given. The newly encountered name requires a `window` size to be given", name));
            let mut moving_avg = MovingAvgLog { sum: 0.0, values: VecDeque::with_capacity(window), window };
            moving_avg.sum += value;
            moving_avg.values.push_back(value);
            self.moving_avg.insert(name, moving_avg);
        }
    }

    /// push a new log of given `ToLog` trait struct
    /// # Warning
    /// - the `window` is ignored after the first encounter of given `name`
    /// # Panic
    /// - panics when a given `name` was seen for the first time and the `window` is None
    pub fn push(&mut self, data: &impl ToLog, window: Option<usize>) {
        let map = data.to_log();
        for (s, v) in map {
            self.push_single(s, v, window);
        }
    }

    /// give the moving average from the slot of given name
    /// # Panic
    /// panics if given name was not registered
    pub fn emit(&self, name: &'static str) -> f32 {
        let moving_avg = self.moving_avg.get(name);
        if moving_avg.is_some() {
            let moving_avg = moving_avg.unwrap();
            moving_avg.sum / moving_avg.values.len() as f32
        } else {
            eprintln!("[MovingAvgLogger]: given name {name} was not registered. Registered names are:");
            for (name, _) in &self.moving_avg {
                eprintln!("{name}")
            }
            f32::NAN
        }
    }
}

/// convert the object to log
pub trait ToLog {
    /// convert itself to a pair of string and value
    fn to_log(&self) -> HashMap<&'static str, f32>;
}

#[cfg(test)]
mod tests {
    use crate::logger::MovingAvgLogger;

    #[test]
    fn register_test() {
        let mut logger = MovingAvgLogger::new();
        logger.register("log1", 10);
        logger.moving_avg.get("log1").expect("SHOULD NOT PANIC");
    }

    #[test]
    fn moving_avg_test() {
        let mut logger = MovingAvgLogger::new();
        logger.register("log1", 10);
        let answers = [0f32, 1.0 / 2.0, 3.0 / 3.0, 6.0 / 4.0, 10.0 / 5.0, 15.0 / 6.0, 21.0 / 7.0, 28.0 / 8.0, 36.0 / 9.0, 45.0 / 10.0];
        for i in 0..10 {
            logger.push_single("log1", i as f32, None);
            assert!(logger.emit("log1") == answers[i]); 
        }

        logger.push_single("log1", 100.0, None);
        let answer = 14.5f32;
        assert!(logger.emit("log1") == answer);

        logger.push_single("log1", 200.0, None);
        assert!((logger.emit("log1") - 34.4).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "The given name unregis was unregistered, but the `window` size was not given. The newly encountered name requires a `window` size to be given")]
    fn push_test() {
        let mut logger = MovingAvgLogger::new();
        logger.register("regis", 10);
        logger.push_single("unregis", 10.0, None);
    }

    #[test]
    fn unregistered_emit_test() {
        let mut logger = MovingAvgLogger::new();
        logger.register("regis", 10);
        logger.emit("unregis");
    }
}