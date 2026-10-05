//! A cliff walking environment for tabular rl algorithms
//! 

use crate::{constraint::Unconstrained, env::Environment};
use burn::prelude::*;
/// A 4 * 12 cliff walking environment implementation
/// - The algorithm has to find an efficient way to reach the goal.
/// - The start position is (0, 0), and the goal is at (11, 0). There are cliffs along (1 .. 10, 0)
/// - Every step gives agent a reward of -1, except when the agent reach the goal and receives reward of 100.
/// - When the agent meets a cliff, the agent is moved back to the start position with reward of -100
pub struct CliffWalking {
    pos: (usize, usize),
    device: Device,
}

impl CliffWalking {
    /// create a new `CliffWalking` environment with start position (0, 0)
    pub fn new(device: &Device) -> Self {
        Self { pos: (0, 0), device: device.clone() }
    }

    fn pos2usize(&self) -> usize {
        self.pos.1 * 12 + self.pos.0
    }

    /// return the number of states
    pub fn n_obs(&self) -> usize { 48 }

    /// return the number of actions
    pub fn n_actions(&self) -> usize { 4 }
}

impl Environment for CliffWalking {
    type Obs = Tensor<2>;
    type Action = Tensor<1, Int>;
    type Constraint = Unconstrained;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        self.pos = (0, 0);
        let mut obs = vec![0f32; self.n_obs()];
        obs[self.pos2usize()] = 1f32;
        (Tensor::<1>::from_floats(obs.as_slice(), &self.device).unsqueeze_dim(0), Unconstrained)
    }

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        let action = action.into_scalar();
        assert!(action == 0 || action == 1 || action == 2 || action == 3);
        let mut next_pos: (isize, isize) = (self.pos.0 as isize, self.pos.1 as isize);

        match action {
            0 => next_pos.1 -= 1,
            1 => next_pos.1 += 1,
            2 => next_pos.0 -= 1,
            3 => next_pos.0 += 1,
            _ => panic!("Invalid action was given")
        }
        let mut reward = -1f32;
        let mut terminated = false;
        
        if next_pos.0 < 0 || next_pos.0 > 11 || next_pos.1 < 0 || next_pos.1 > 3 {
            // the agent is out of boundary
            next_pos = (self.pos.0 as isize, self.pos.1 as isize);
        } else if next_pos.1 == 0 && (0 < next_pos.0 && next_pos.0 < 11) {
            // the agent met a cliff
            next_pos = (0, 0);
            reward = -100f32;
        } else if next_pos == (11, 0) {
            // the agent met a goal
            reward = 100f32;
            terminated = true
        }

        self.pos = (next_pos.0 as usize, next_pos.1 as usize);
        let mut obs = vec![0f32; self.n_obs()];
        obs[self.pos2usize()] = 1f32;
        ((Tensor::<1>::from_floats(obs.as_slice(), &self.device).unsqueeze_dim(0), Unconstrained), reward, terminated, false)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}
