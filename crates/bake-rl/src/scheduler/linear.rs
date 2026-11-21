use super::Scheduler;
use std::ops::{Div, Add, Mul, Sub};

/// linear scheduler
pub struct LinearScheduler<T: Add<Output = T> + Sub<Output = T> + Div<f32, Output = T> + Mul<f32, Output = T> + Clone> {
    start: T,
    end: T,
    slope: T,
    steps: usize,
}

impl<T: Add<Output = T> + Sub<Output = T> + Div<f32, Output = T> + Mul<f32, Output = T> + Clone> LinearScheduler<T> {
    /// create new `LinearScheduler`
    pub fn new(start: T, end: T, steps: usize, fraction: f32) -> Self {
        assert!(0. <= fraction && fraction <= 1., "fraction out of range");
        let steps = (steps as f32 * fraction) as usize;
        let slope = (end.clone() - start.clone()) / steps as f32;
        Self {
            start,
            end,
            slope,
            steps,
        }
    }
}

impl<T: Add<Output = T> + Sub<Output = T> + Div<f32, Output = T> + Mul<f32, Output = T> + Clone> Scheduler for LinearScheduler<T> {
    type Scheduled = T;

    fn value(&self, step: usize) -> Self::Scheduled {
        if step <= self.steps {
            self.slope.clone() * step as f32 + self.start.clone()
        } else {
            self.end.clone()
        }
    }
}