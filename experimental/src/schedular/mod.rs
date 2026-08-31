
pub struct LinearSchedular {
    start: f32,
    end: f32,
    steps: usize,

    cur_step: usize,
    slope: f32,
}

impl LinearSchedular {
    pub fn new(start: f32, end: f32, steps: usize) -> Self {
        Self {
            start,
            end,
            steps,
            cur_step: 0,
            slope: (end - start) / steps as f32
        }
    }

    pub fn step(&mut self) -> f32 {
        if self.cur_step <= self.steps { self.cur_step += 1; }
        let ret = self.slope * self.cur_step as f32 + self.start;
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
        let s = 0f32;
        let e = 1f32;
        let total_steps = 10000;
        let mut sch = LinearSchedular::new(s, e, total_steps);
        let mut observed = 0f32;
        for _ in 0..total_steps / 2 { observed = sch.step(); }
        assert!((observed - 0.5f32).abs() < 5e-3, "got {observed}, expected: 0.5");
        for _ in 0..total_steps / 2 { observed = sch.step(); }
        assert!((observed - 1.0f32).abs() < 5e-3, "got {observed}, expected: 1.0");
    }
}