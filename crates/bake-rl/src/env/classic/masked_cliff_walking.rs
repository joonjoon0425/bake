//! Cliff Walking with masks at the boundaries
use burn::prelude::*;
use crate::core::{env::{Environment, EnvironmentConfig, space::DiscreteSpace}};
/// CliffWalking with mask implementation
/// - All feature are same but on the boundary, the agent receives an mask.
/// - The agent won't be able to go left from the left boundary, right at right boundary, and so on.
pub struct MaskedCliffWalking {
    pos: (usize, usize),
    obs_space: DiscreteSpace,
    action_space: DiscreteSpace,
    device: Device,
}

impl MaskedCliffWalking {

    fn pos2tensor(&self) -> Tensor<1, Int> {
        Tensor::from_ints([self.pos.1 * 12 + self.pos.0], &self.device)
    }
}

impl Environment for MaskedCliffWalking {
    type Constraint = Tensor<2, Bool>;
    type ObsSpace = DiscreteSpace;
    type ActionSpace = DiscreteSpace;
    type Reward = Tensor<1>;
    type Info = ();

    fn reset(&mut self) -> ((Tensor<1, Int>, Self::Constraint), Self::Info) {
        self.pos = (0, 0);
        ((self.pos2tensor(), Tensor::from_bool([false, true, false, true], &self.device)), ())
    }

    fn step(&mut self, action: Tensor<1, Int>) -> ((Tensor<1, Int>, Self::Constraint), Tensor<1>, bool, bool, ()) {
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
        ((self.pos2tensor(), Tensor::from_bool(mask, &self.device)), Tensor::from_floats([reward], &self.device), terminated, false, ())
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn device(&self) -> Device {
        self.device.clone()
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
    fn init(self, device: &Device) -> Self::Environment {
        let obs_space = DiscreteSpace::new(48, device);
        let action_space = DiscreteSpace::new(4, device);
        Self::Environment { pos: (0, 0), obs_space, action_space, device: device.clone() }
    }
}