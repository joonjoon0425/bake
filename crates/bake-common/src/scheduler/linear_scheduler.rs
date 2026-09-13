//! A Linear scheduler implementation
//! 
use crate::scheduler::Scheduler;

/// A liner scheduler implementation
pub struct LinearScheduler {
    start: f64,
    cur_step: usize,
    steps: usize,
    slope: f64,
}

impl LinearScheduler {
    /// create a new `LinearScheduler`
    /// - `start`: the value which scheduled value starts from
    /// - `end`: the value which scheduled value ends at
    /// - `steps`: total steps of the outer loop
    /// - `fraction`: the scheduler will schedule from `start` to `end` for `steps` * `fraction` steps
    /// ```
    /// use bake_common::scheduler::{Scheduler, LinearScheduler};
    /// let mut sch = LinearScheduler::new(1.0, 0.05, 100, 0.5);
    /// let mut v = 1.0f64;
    /// for _ in 0..50 { v = sch.step(); }
    /// assert!((v - 0.05).abs() < 1e-8);
    /// ```
    pub fn new(start: f64, end: f64, steps: usize, fraction: f64) -> Self {
        assert!(0.0 < fraction && fraction <= 1.0);
        let steps = (steps as f64 * fraction) as usize;
        Self {
            start,
            steps,
            cur_step: 0,
            slope: (end - start) / steps as f64
        }
    }
}

impl Scheduler for LinearScheduler {
    type Scheduled = f64;

    fn step(&mut self) -> f64 {
        if self.cur_step < self.steps { self.cur_step += 1; }
        let ret = self.slope * self.cur_step as f64 + self.start;
        ret
    }

    fn reset(&mut self) {
        self.cur_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::scheduler::LinearScheduler;

    #[test]
    #[should_panic]
    fn wrong_fraction_below_zero() {
        LinearScheduler::new(1.0, 0.05, 100, -1.0);
    }

    #[test]
    #[should_panic]
    fn wrong_fraction_over_one() {
        LinearScheduler::new(1.0, 0.05, 100, 20.0);
    }
}