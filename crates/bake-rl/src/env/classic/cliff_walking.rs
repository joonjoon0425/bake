//! Cliff Walking
use crate::env::space::{Space, DiscreteSpace};
use crate::env::{Environment, EnvironmentConfig};

/// A 4 * 12 cliff walking environment implementation
/// - The algorithm has to find an efficient way to reach the goal.
/// - The start position is (0, 0), and the goal is at (11, 0). There are cliffs along (1 .. 10, 0)
/// - Every step gives agent a reward of -1, except when the agent reach the goal and receives reward of 100.
/// - When the agent meets a cliff, the agent is moved back to the start position with reward of -100
pub struct CliffWalking {
    pos: (usize, usize),
    obs_space: DiscreteSpace,
    action_space: DiscreteSpace,
}

impl CliffWalking {
    fn pos2usize(&self) -> usize { self.pos.1 * 12 + self.pos.0 }
}

impl Environment for CliffWalking {
    type Constraint = ();
    type ObsSpace = DiscreteSpace;
    type ActionSpace = DiscreteSpace;
    type Info = ();
    type Reward = f32;

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), ()) {
        self.pos = (0, 0);
        ((self.pos2usize(), ()), ())
    }

    fn step(&mut self, action: usize) -> ((usize, Self::Constraint), f32, bool, bool, ()) {
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
        ((self.pos2usize(), ()), reward, terminated, false, ())
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }
}

/// configuration for cliffwalking
#[derive(Debug, Clone, Copy)]
pub struct CliffWalkingConfig;

impl EnvironmentConfig for CliffWalkingConfig {
    type Environment = CliffWalking;
    fn init(self) -> Self::Environment {
        let obs_space = DiscreteSpace::new(48);
        let action_space = DiscreteSpace::new(4);
        Self::Environment { pos: (0, 0), obs_space, action_space }
    }
}