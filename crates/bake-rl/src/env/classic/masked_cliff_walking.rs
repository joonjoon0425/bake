//! Cliff Walking with masks at the boundaries
use crate::core::env::{Environment, EnvironmentConfig, space::DiscreteSpace};
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
    type Constraint = Vec<bool>;
    type ObsSpace = DiscreteSpace;
    type ActionSpace = DiscreteSpace;
    type Reward = f32;
    type Info = ();

    fn reset(&mut self) -> ((usize, Self::Constraint), Self::Info) {
        self.pos = (0, 0);
        ((self.pos2usize(), vec![false, true, false, true]), ())
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
        let mut mask = vec![true; 4];
        
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
pub struct MaskedCliffWalkingConfig;
impl MaskedCliffWalkingConfig {
    /// create a new configuration
    pub fn new() -> Self { Self }
}

impl EnvironmentConfig for MaskedCliffWalkingConfig {
    type Environment = MaskedCliffWalking;
    fn init(self) -> Self::Environment {
        let obs_space = DiscreteSpace::new(48);
        let action_space = DiscreteSpace::new(4);
        Self::Environment { pos: (0, 0), obs_space, action_space }
    }
}



#[cfg(test)]
mod tests {
    use crate::core::env::{Environment, EnvironmentConfig};
    use crate::env::classic::masked_cliff_walking::MaskedCliffWalkingConfig;

    #[test]
    #[should_panic]
    fn invalid_action() {
        let mut env = MaskedCliffWalkingConfig::new().init();
        env.reset();
        env.step(5);
    }

    #[test]
    fn udlr() {
        let mut env = MaskedCliffWalkingConfig::new().init();
        env.reset();
        let ((pos, _), _, _, _, _) = env.step(1);
        assert_eq!(pos, 12);
        let ((pos, _), _, _, _, _) = env.step(3);
        assert_eq!(pos, 13);
        let ((pos, _), _, _, _, _) = env.step(1);
        assert_eq!(pos, 25);
    }

    #[test]
    fn cliff_to_start() {
        let mut env = MaskedCliffWalkingConfig::new().init();
        env.reset();
        let ((pos, _), _, _, _, _) = env.step(3);
        assert_eq!(pos, 0)
    }

    #[test]
    fn goal_terminate_reward() {
        let mut env = MaskedCliffWalkingConfig::new().init();
        env.reset();

        let (pos, reward, terminated, _, _) = env.step(1); // (0, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _, _) = env.step(3); // (1, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _, _) = env.step(3); // (2, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _, _) = env.step(3); // (3, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (pos, reward, terminated, _, _) = env.step(3); // (4, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl on {}", pos.0) }
        let (_, reward, terminated, _, _) = env.step(3); // (5, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3); // (6, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3); //
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3);
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(3); // (11, 1)
        if reward != -1f32 || terminated { panic!("Wrong env impl") }
        let (_, reward, terminated, _, _) = env.step(0); // (11, 0) goal
        if reward != 100f32 || !terminated { panic!("Wrong env impl") }
    }

    // #[test]
    // fn masking() {
    //     let qtable = QTableConfig::new(48, 4).init();
    //     let mut exploration = EpsGreedy::new(1, 1.0);

    //     let mut env = MaskedCliffWalkingConfig::new().init();
    //     let (obs, constraint) = env.reset();

    //     for _ in 0..1000 {
    //         let action = exploration.sample(qtable.qvalues_as_vec(obs), constraint);
    //         assert!(action != 2 && action != 0);
    //     }
    // }
}