use super::Scheduler;

/// exponential scheduler
pub struct ExponentialScheduler {
    log_start: f32,
    log_slope: f32,
    log_end: f32,
    steps: usize,
    sign: f32,
}

impl ExponentialScheduler {
    /// create new `ExponentialScheduler`
    pub fn new(start: f32, end: f32, steps: usize, fraction: f32) -> Self {
        assert!(0. <= fraction && fraction <= 1., "fraction out of range");
        assert!(start * end > 0., "`start` and `end` has different sign");
        let sign = start.signum();
        let steps = (steps as f32 * fraction) as usize;
        let log_slope = (end.abs().ln() - start.abs().ln()) / steps as f32;
        Self {
            log_start: start.abs().ln(),
            log_end: end.abs().ln(),
            log_slope,
            steps,
            sign,
        }
    }
}

impl Scheduler for ExponentialScheduler {
    type Scheduled = f32;

    fn value(&self, step: usize) -> Self::Scheduled {
        if step <= self.steps {
            self.sign * (self.log_start + step as f32 * self.log_slope).exp()
        } else {
            self.log_end.exp() * self.sign
        }
    }
}