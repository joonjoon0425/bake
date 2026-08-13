use crate::env::*;

pub struct GridWorld {
    s: (i32, i32)
}

impl GridWorld {
    pub fn new() -> Self { Self { s: (0, 0) } }
    pub fn n_states(&self) -> usize {
        105
    }
    pub fn n_actions(&self) -> usize {
        4
    }
}

impl Env for GridWorld {
    type Mask = NoMask<4>;

    fn reset(&mut self) -> usize {
        self.s = (0, 0);
        0
    }

    fn step(&mut self, action: usize) -> Step {
        let mut pos = self.s;
        match action {
            0 => pos.1 -= 1,
            1 => pos.1 += 1,
            2 => pos.0 += 1,
            3 => pos.0 -= 1,
            _ => panic!("Invalid action given."),
        }
        
        let (reward, done) = if pos.0 < 0 || pos.0 > 14 {
            pos = (0, 0);
            (-100f32, false)
        } else if pos.1 < 0 || pos.1 > 6 {
            pos = (0, 0);
            (-100f32, false)
        } else if pos == (14, 0) {
            (100f32, true)
        } else {
            (-1f32, false)
        };

        self.s = pos;

        return Step {
            obs: (pos.0 + pos.1 * 15) as usize,
            reward,
            done,
            truncated: false
        };
    }

    fn action_mask(&self) -> Self::Mask {
        NoMask
    }

}

