use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::environment::Environment;

/// Gymnasium CartPole-v1과 동일한 역학.
pub struct CartPole {
    state: [f32; 4],
    steps: usize,
    rng: StdRng,
}

impl CartPole {
    const GRAVITY: f32 = 9.8;
    const MASS_CART: f32 = 1.0;
    const MASS_POLE: f32 = 0.1;
    const TOTAL_MASS: f32 = Self::MASS_CART + Self::MASS_POLE;
    const LENGTH: f32 = 0.5;
    const POLE_MASS_LENGTH: f32 = Self::MASS_POLE * Self::LENGTH;
    const FORCE_MAG: f32 = 10.0;
    const TAU: f32 = 0.02;

    const THETA_THRESHOLD: f32 = 12.0 * std::f32::consts::PI / 180.0;
    const X_THRESHOLD: f32 = 2.4;
    const MAX_STEPS: usize = 500;

    pub const OBS_DIM: usize = 4;
    pub const N_ACTIONS: usize = 2;

    pub fn new(seed: u64) -> Self {
        Self {
            state: [0.0; 4],
            steps: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Environment for CartPole {
    type Obs = [f32; 4];
    type Action = i64;

    fn reset(&mut self) -> [f32; 4] {
        for v in self.state.iter_mut() {
            *v = (self.rng.random::<f32>() - 0.5) * 0.1; // U(-0.05, 0.05)
        }
        self.steps = 0;
        self.state
    }

    /// action: 0 = 왼쪽, 1 = 오른쪽
    fn step(&mut self, action: i64) -> (Self::Obs, f32, bool, bool) {
        let [x, x_dot, theta, theta_dot] = self.state;

        let force = if action == 1 { Self::FORCE_MAG } else { -Self::FORCE_MAG };
        let cos = theta.cos();
        let sin = theta.sin();

        let temp = (force + Self::POLE_MASS_LENGTH * theta_dot * theta_dot * sin)
            / Self::TOTAL_MASS;
        let theta_acc = (Self::GRAVITY * sin - cos * temp)
            / (Self::LENGTH * (4.0 / 3.0 - Self::MASS_POLE * cos * cos / Self::TOTAL_MASS));
        let x_acc = temp - Self::POLE_MASS_LENGTH * theta_acc * cos / Self::TOTAL_MASS;

        self.state[0] = x + Self::TAU * x_dot;
        self.state[1] = x_dot + Self::TAU * x_acc;
        self.state[2] = theta + Self::TAU * theta_dot;
        self.state[3] = theta_dot + Self::TAU * theta_acc;

        self.steps += 1;

        let terminated = self.state[0].abs() > Self::X_THRESHOLD
            || self.state[2].abs() > Self::THETA_THRESHOLD;
        let truncated = !terminated && self.steps >= Self::MAX_STEPS;

        
        (self.state, 1.0, terminated, truncated)
        
    }
}