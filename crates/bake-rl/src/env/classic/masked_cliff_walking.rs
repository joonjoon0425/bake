//! Cliff Walking with masks at the boundaries
use crate::env::space::{DiscreteSpace, Space};
use crate::env::{Environment, EnvironmentConfig};

/// CliffWalking with mask implementation
/// - All feature are same but on the boundary, the agent receives an mask.
/// - The agent won't be able to go left from the left boundary, right at right boundary, and so on.
pub struct MaskedCliffWalking {
    pos: (usize, usize),
    obs_space: DiscreteSpace,
    action_space: DiscreteSpace,
}

impl MaskedCliffWalking {

    fn pos2usize(&self) -> usize {
        self.pos.1 * 12 + self.pos.0
    }
}

impl Environment for MaskedCliffWalking {
    type Constraint = [bool; 4];
    type ObsSpace = DiscreteSpace;
    type ActionSpace = DiscreteSpace;
    type Reward = f32;
    type Info = ();

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        self.pos = (0, 0);
        ((self.pos2usize(), [false, true, false, true]), ())
    }

    fn step(&mut self, action: usize) -> ((usize, Self::Constraint), f32, bool, bool, ()) {
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
        let mut mask = [true; 4];
        
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
        ((self.pos2usize(), mask), reward, terminated, false, ())
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }
}

/// configuration for masked cliffwalking
#[derive(Debug, Clone, Copy)]
pub struct MaskedCliffWalkingConfig;

impl EnvironmentConfig for MaskedCliffWalkingConfig {
    type Environment = MaskedCliffWalking;
    fn init(self) -> Self::Environment {
        let obs_space = DiscreteSpace::new(48);
        let action_space = DiscreteSpace::new(4);
        Self::Environment { pos: (0, 0), obs_space, action_space }
    }
}