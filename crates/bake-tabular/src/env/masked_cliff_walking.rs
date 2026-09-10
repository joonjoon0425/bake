//! A cliff walking which are masked on boundaries
//! 

use crate::{constraint::DiscreteMask, env::Environment};
/// CliffWakling with mask implementation
/// - All feature are same but on the boundary, the agent receives an mask.
/// - The agent won't be able to go left from the left boundary, right at right boundary, and so on.
pub struct MaskedCliffWalking {
    pos: (usize, usize)
}

impl MaskedCliffWalking {
    /// create a new `MaskedCliffWalking` environment with start position (0, 0)
    pub fn new() -> Self { Self { pos: (0, 0) } }

    /// return the number of states
    pub fn n_states() -> usize { 48 }

    /// return the number of actions
    pub fn n_actions() -> usize { 4 }

    fn pos2usize(&self) -> usize {
        self.pos.1 * 12 + self.pos.0
    }
}

impl Environment for MaskedCliffWalking {
    type Constraint = DiscreteMask<4>;

    fn reset(&mut self) -> (usize, Self::Constraint) {
        self.pos = (0, 0);
        (self.pos2usize(), DiscreteMask::from_bool([false, true, false, true]))
    }

    fn step(&mut self, action: usize) -> ((usize, Self::Constraint), f32, bool, bool) {
        assert!(action == 0 || action == 1 || action == 2 || action == 3);
        let mut next_pos: (usize, usize) = self.pos;

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
        let mut mask = DiscreteMask::new(true);
        
        // masking
        if next_pos.0 == 0 {
            mask.disable(2);
        } else if next_pos.0 == 11 {
            mask.disable(3);
        }

        if next_pos.1 == 0 {
            mask.disable(0);
        } else if next_pos.1 == 3 {
            mask.disable(1);
        }
        
        if next_pos.1 == 0 && (0 < next_pos.0 && next_pos.0 < 11) {
            // the agent met a cliff
            next_pos = (0, 0);
            reward = -100f32;
        } else if next_pos == (11, 0) {
            // the agent met a goal
            reward = 100f32;
            terminated = true
        }

        self.pos = (next_pos.0 as usize, next_pos.1 as usize);
        ((self.pos2usize(), mask), reward, terminated, false)
    }
}


#[cfg(test)]
mod tests {
    use crate::{env::{Environment, MaskedCliffWalking}, explore::{EpsGreedy, Exploration}, qtable::QTable};

    #[test]
    #[should_panic]
    fn invalid_action() {
        let mut env = MaskedCliffWalking::new();
        env.reset();
        env.step(5);
    }

    #[test]
    fn udlr() {
        let mut env = MaskedCliffWalking::new();
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
        let mut env = MaskedCliffWalking::new();
        env.reset();
        let ((pos, _), _, _, _) = env.step(3);
        assert_eq!(pos, 0)
    }

    #[test]
    fn goal_terminate_reward() {
        let mut env = MaskedCliffWalking::new();
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

    #[test]
    fn masking() {
        let qtable = QTable::new(48, 4);
        let mut exploration = EpsGreedy::new(1, 1.0);

        let mut env = MaskedCliffWalking::new();
        let (obs, constraint) = env.reset();

        for _ in 0..1000 {
            let action = exploration.sample(&qtable, obs, constraint);
            assert!(action != 2 && action != 0);
        }
    }
}