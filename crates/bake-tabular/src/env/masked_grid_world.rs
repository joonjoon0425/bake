//! ## Masked GridWorld
//! This is an implementation of gridworld with mask
//! 

use crate::env::*;

/// ### An implementation of gridworld
/// - Size 15 * 7, starts at (0, 0) and the goal is (14, 0)
/// - Cliffs exists along (1, 0), (2, 0), ... , (13, 0)
/// - actions: 0 - up, 1 - down, 2 - left, 3 - right
/// - terminates when the agent meets a cliff or goal
pub struct MaskedGridWorld {
    s: (i32, i32)
}

impl MaskedGridWorld {
    /// Create a new GridWorld
    pub fn new() -> Self { Self { s: (0, 0) } }
    /// number of states in GridWorld
    pub fn n_states(&self) -> usize {
        105
    }
    /// number of actions in GridWorld
    pub fn n_actions(&self) -> usize {
        4
    }

    fn action_mask((x, y): (i32, i32)) -> <Self as Env>::Mask {
        let mut mask = <Self as Env>::Mask::new(true);
        
        if x <= 0 {
            mask.disable(2);
        } else if x >= 14 {
            mask.disable(3);
        }

        if y <= 0 {
            mask.disable(0);
        } else if y >= 6 {
            mask.disable(1);
        }

        mask
    }
}

impl Env for MaskedGridWorld {
    type Mask = DiscreteMask<4>;

    fn reset(&mut self) -> (usize, Self::Mask) {
        self.s = (0, 0);
        (0, Self::action_mask(self.s))
    }

    fn step(&mut self, action: usize) -> (usize, f32, bool, bool, Self::Mask) {
        let mut pos = self.s;
        match action {
            0 => pos.1 -= 1,
            1 => pos.1 += 1,
            2 => pos.0 -= 1,
            3 => pos.0 += 1,
            _ => panic!("Invalid action given."),
        }
        
        let (reward, done) = if pos.1 == 0 && (0 < pos.0 && pos.0 < 14) {
            (-100f32, true)
        } else if pos == (14, 0) {
            (100f32, true)
        } else {
            (-1f32, false)
        };

        self.s = pos;

        ((pos.0 + pos.1 * 15) as usize, reward, done, false, Self::action_mask(pos))
    }
}

