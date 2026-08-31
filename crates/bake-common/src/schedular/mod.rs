//! A Schedular for various scalar values
pub struct LinearSchedular {
    start: f64,
    steps: usize,

    cur_step: usize,
    slope: f64,
}

impl LinearSchedular {
    pub fn new(start: f64, end: f64, steps: usize) -> Self {
        Self {
            start,
            steps,
            cur_step: 0,
            slope: (end - start) / steps as f64
        }
    }

    pub fn step(&mut self) -> f64 {
        if self.cur_step < self.steps { self.cur_step += 1; }
        let ret = self.slope * self.cur_step as f64 + self.start;
        ret
    }

    pub fn reset(&mut self) {
        self.cur_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::LinearSchedular;

    #[test]
    pub fn check_linear_schduling() {
        let s = 0.;
        let e = 1.;
        let total_steps = 10000;
        let mut sch = LinearSchedular::new(s, e, total_steps);
        let mut observed = 0.;
        for _ in 0..total_steps / 2 { observed = sch.step(); }
        assert!((observed - 0.5).abs() < 5e-3, "got {observed}, expected: 0.5");
        for _ in 0..total_steps / 2 { observed = sch.step(); }
        assert!((observed - 1.0).abs() < 5e-3, "got {observed}, expected: 1.0");
    }
}