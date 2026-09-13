//! CartPole-v1 environment.
//!
//! Physics and constants follow Gymnasium's `CartPoleEnv` (Euler integrator).
//! The action space is discrete with two actions (`0` = push left, `1` = push right),
//! so the mask is always all-true; it exists only so that mask-aware networks
//! (e.g. dueling) can be exercised on an unmasked task.
 
use burn::{
    Tensor, tensor::{Device, Int},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::{constraint::Unconstrained, env::Environment};
 
const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
/// Half the pole's length, as in Gymnasium.
const LENGTH: f32 = 0.5;
const POLEMASS_LENGTH: f32 = MASS_POLE * LENGTH;
const FORCE_MAG: f32 = 10.0;
/// Seconds between state updates.
const TAU: f32 = 0.02;
 
/// Termination threshold on the pole angle (12 degrees, in radians).
const THETA_THRESHOLD: f32 = 12.0 * core::f32::consts::PI / 180.0;
/// Termination threshold on the cart position.
const X_THRESHOLD: f32 = 2.4;
/// Truncation limit for the `-v1` variant.
const MAX_STEPS: u32 = 500;
 
/// Range of the uniform distribution used to initialise every state variable.
const INIT_RANGE: f32 = 0.05;
 
/// The number of discrete actions.
pub const N_ACTIONS: usize = 2;
/// The dimensionality of an observation.
pub const OBS_DIM: usize = 4;
 
/// The classic cart-pole balancing task.
///
/// Observation is `[x, x_dot, theta, theta_dot]` as a `Tensor<1>` of shape `[4]`.
/// Reward is `1.0` on every step, including the terminating one.
pub struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    steps: u32,
    rng: StdRng,
    device: Device,
    /// Pre-built all-true mask. Cloning a tensor clones a handle, not the buffer,
    /// so this avoids rebuilding it on every step.
    mask: <Self as Environment>::Constraint,
}
 
impl CartPole {
    /// Create a new environment with a fixed seed.
    ///
    /// The environment is not usable until [`Env::reset`] has been called.
    pub fn new(seed: u64, device: &Device) -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            rng: StdRng::seed_from_u64(seed),
            device: device.clone(),
            mask: Unconstrained,
        }
    }
 
    /// Build the observation tensor from the current state.
    fn obs(&self) -> Tensor<2> {
        Tensor::<2>::from_data(
            [[self.x, self.x_dot, self.theta, self.theta_dot]],
            &self.device,
        )
    }
 
    /// Whether the current state is outside the failure thresholds.
    fn is_terminal(&self) -> bool {
        self.x.abs() > X_THRESHOLD || self.theta.abs() > THETA_THRESHOLD
    }
}
 
impl Environment for CartPole {
    type Obs = Tensor<2>;
    type Action = Tensor<1, Int>;
    type Constraint = Unconstrained;
 
    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        let mut sample = || self.rng.random_range(-INIT_RANGE..INIT_RANGE);
 
        self.x = sample();
        self.x_dot = sample();
        self.theta = sample();
        self.theta_dot = sample();
        self.steps = 0;
 
        (self.obs(), self.mask.clone())
    }
 
    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        let action = action.into_scalar();
        debug_assert!(
            (0..N_ACTIONS as i64).contains(&action),
            "action out of range: {action}"
        );
 
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let (sin_theta, cos_theta) = self.theta.sin_cos();
 
        let temp =
            (force + POLEMASS_LENGTH * self.theta_dot * self.theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;
 
        // Euler integration; the ordering matters and matches Gymnasium.
        self.x += TAU * self.x_dot;
        self.x_dot += TAU * x_acc;
        self.theta += TAU * self.theta_dot;
        self.theta_dot += TAU * theta_acc;
 
        self.steps += 1;
 
        let terminated = self.is_terminal();
        let truncated = !terminated && self.steps >= MAX_STEPS;
 
        (
            (self.obs(), self.mask.clone()),
            1.0,
            terminated,
            truncated,
        )
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}
