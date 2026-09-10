//! A cliff walking environment for tabular rl algorithms
//! 

use crate::{constraint::Unconstrained, env::Environment};

/// A 4 * 12 cliff walking environment implementation
/// - The algorithm has to find an efficient way to reach the goal.
/// - The start position is (0, 0), and the goal is at (11, 0). There are cliffs along (1 .. 10, 0)
/// - Every step gives agent a reward of -1, except when the agent reach the goal and receives reward of 100.
/// - When the agent meets a cliff, the agent is moved back to the start position with reward of -100
pub struct CliffWalking {
    pos: (usize, usize)
}

impl CliffWalking {
    /// create a new `CliffWalking` environment with start position (0, 0)
    pub fn new() -> Self {
        Self { pos: (0, 0) }
    }

    fn pos2usize(&self) -> usize {
        self.pos.1 * 12 + self.pos.0
    }

    /// return the number of states
    pub fn n_states() -> usize { 48 }

    /// return the number of actions
    pub fn n_actions() -> usize { 4 }
}

impl Environment for CliffWalking {
    type Constraint = Unconstrained<4>;

    fn reset(&mut self) -> (usize, Self::Constraint) {
        self.pos = (0, 0);
        (self.pos2usize(), Unconstrained)
    }

    fn step(&mut self, action: usize) -> ((usize, Self::Constraint), f32, bool, bool) {
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
        ((self.pos2usize(), Unconstrained), reward, terminated, false)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{CliffWalking, Environment};

    #[test]
    #[should_panic]
    fn invalid_action() {
        let mut env = CliffWalking::new();
        env.reset();
        env.step(5);
    }

    #[test]
    fn udlr() {
        let mut env = CliffWalking::new();
        env.reset();
        let ((pos, _), _, _, _) = env.step(1);
        assert_eq!(pos, 12);
        let ((pos, _), _, _, _) = env.step(3);
        assert_eq!(pos, 13);
        let ((pos, _), _, _, _) = env.step(1);
        assert_eq!(pos, 25);
    }

    #[test]
    fn cliff_to_start() {
        let mut env = CliffWalking::new();
        env.reset();
        let ((pos, _), _, _, _) = env.step(3);
        assert_eq!(pos, 0)
    }

    #[test]
    fn goal_terminate_reward() {
        let mut env = CliffWalking::new();
        env.reset();

        let (pos, reward, terminated, _) = env.step(1); // (0, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _) = env.step(3); // (1, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _) = env.step(3); // (2, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _) = env.step(3); // (3, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _) = env.step(3); // (4, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (_, reward, terminated, _) = env.step(3); // (5, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3); // (6, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3); //
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(3); // (11, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _) = env.step(0); // (11, 0) goal
        if reward != 100f32 || !terminated { panic!("Wrong env impl") }
    }
}