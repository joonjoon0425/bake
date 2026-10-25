//! CartPole-v1 environment.
//!
//! Physics and constants follow Gymnasium's `CartPoleEnv` (Euler integrator).
//! The action space is discrete with two actions (`0` = push left, `1` = push right),
//! so the mask is always all-true; it exists only so that mask-aware networks
//! (e.g. dueling) can be exercised on an unmasked task.
use burn::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use crate::core::constraint::Unconstrained;
use crate::core::env::space::ContinuousSpace;
use crate::core::env::space::DiscreteSpace;
use crate::core::env::space::Space;

use crate::core::env::Environment;
use crate::core::env::EnvironmentConfig;
 
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
    /// Pre-built all-true mask. Cloning a tensor clones a handle, not the buffer,
    /// so this avoids rebuilding it on every step.
    mask: <Self as Environment>::Constraint,

    device: Device,

    obs_space: ContinuousSpace<2>,
    action_space: DiscreteSpace,
}
 
impl CartPole {
    /// Build the observation tensor from the current state.
    fn obs(&self) -> Tensor<2> {
        Tensor::from_floats([[self.x, self.x_dot, self.theta, self.theta_dot]], &self.device)
    }
 
    /// Whether the current state is outside the failure thresholds.
    fn is_terminal(&self) -> bool {
        self.x.abs() > X_THRESHOLD || self.theta.abs() > THETA_THRESHOLD
    }
}
 
impl Environment for CartPole {
    type ObsSpace = ContinuousSpace<2>;
    type ActionSpace = DiscreteSpace;
    type Constraint = Unconstrained;
    type Reward = f32;
    type Info = ();

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), ()) {
        let mut sample = || self.rng.random_range(-INIT_RANGE..INIT_RANGE);
 
        self.x = sample();
        self.x_dot = sample();
        self.theta = sample();
        self.theta_dot = sample();
        self.steps = 0;
 
        ((self.obs(), self.mask.clone()), ())
    }
 
    fn step(&mut self, action: Tensor<1, Int>) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), f32, bool, bool, ()) {
        let action = action.into_scalar::<u64>() as usize;
        debug_assert!(
            (0..N_ACTIONS).contains(&action),
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
            ()
        )
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

/// configuration struct for `CartPole` environment
/// # Configurations
/// - seed: seed for randomness. 12 for default
pub struct CartPoleConfig {
    seed: Option<u64>
}

impl CartPoleConfig {
    /// create a new `CartPoleConfig`
    pub fn new() -> Self {
        Self {
            seed: None
        }
    }
    /// set seed for `CartPole`
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    } 
}

impl EnvironmentConfig for CartPoleConfig {
    type Environment = CartPole;
    fn init(self, device: &Device) -> Self::Environment {
        let obs_space = ContinuousSpace::new(&[4], Tensor::from_floats([[-f32::INFINITY; 4]], device), Tensor::from_floats([[f32::INFINITY; 4]], device));
        let action_space = DiscreteSpace::new(2);
        CartPole {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            rng: StdRng::seed_from_u64(*self.seed.as_ref().unwrap_or(&12)),
            mask: Unconstrained,
            obs_space,
            action_space,
            device: device.clone()
        }
    }
}
