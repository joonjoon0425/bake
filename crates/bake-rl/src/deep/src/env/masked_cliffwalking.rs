//! A cliff walking which are masked on boundaries
//! 
use crate::{constraint::discrete_constraint::DiscreteMask, env::Environment};
use burn::prelude::*;
/// CliffWakling with mask implementation
/// - All feature are same but on the boundary, the agent receives an mask.
/// - The agent won't be able to go left from the left boundary, right at right boundary, and so on.
pub struct MaskedCliffWalking {
    pos: (usize, usize),
    device: Device,
}

impl MaskedCliffWalking {
    /// create a new `MaskedCliffWalking` environment with start position (0, 0)
    pub fn new(device: &Device) -> Self { Self { pos: (0, 0), device: device.clone() } }

    /// return the number of states
    pub fn n_obs(&self) -> usize { 48 }

    /// return the number of actions
    pub fn n_actions(&self) -> usize { 4 }

    fn pos2usize(&self) -> usize {
        self.pos.1 * 12 + self.pos.0
    }
}

impl Environment for MaskedCliffWalking {
    type Obs = Tensor<2>;
    type Action = Tensor<1, Int>;
    type Constraint = DiscreteMask;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        self.pos = (0, 0);
        let mut obs = vec![0f32; self.n_obs()];
        obs[self.pos2usize()] = 1f32;
        (Tensor::<1>::from_floats(obs.as_slice(), &self.device).unsqueeze_dim(0), DiscreteMask(Tensor::<1, Bool>::from_bool([false, true, false, true], &self.device).unsqueeze_dim(0)))
    }

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        let action = action.into_scalar();
        assert!(action == 0 || action == 1 || action == 2 || action == 3);
        let mut next_pos = (self.pos.0 as isize, self.pos.1 as isize);

        // assumes that the correct actions were given; the environment won't check if the given action was the masked action or not
        match action {
            0 => next_pos.1 -= 1,
            1 => next_pos.1 += 1,
            2 => next_pos.0 -= 1,
            3 => next_pos.0 += 1,
            _ => panic!("Invalid action was given")
        }
        let mut reward = -1f32;
        let mut terminated = false;
        let mut mask = [true, true, true, true];
        
        if next_pos.1 == 0 && (0 < next_pos.0 && next_pos.0 < 11) {
            // the agent met a cliff
            next_pos = (0, 0);
            reward = -100f32;
        } else if next_pos == (11, 0) {
            // the agent met a goal
            reward = 100f32;
            terminated = true
        }

        // masking
        if next_pos.0 <= 0 {
            mask[2] = false;
        } else if next_pos.0 >= 11 {
            mask[3] = false;
        }

        if next_pos.1 <= 0 {
            mask[0] = false;
        } else if next_pos.1 >= 3 {
            mask[1] = false;
        }

        self.pos = (next_pos.0 as usize, next_pos.1 as usize);
        let mut obs = vec![0f32; self.n_obs()];
        obs[self.pos2usize()] = 1f32;
        ((Tensor::<1>::from_floats(obs.as_slice(), &self.device).unsqueeze_dim(0), DiscreteMask(Tensor::<1, Bool>::from_bool(mask, &self.device).unsqueeze_dim(0))), reward, terminated, false)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}