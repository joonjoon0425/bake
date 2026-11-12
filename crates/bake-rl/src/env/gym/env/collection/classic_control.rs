//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    /// `MountainCar-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mountain_car() -> Self {
        Self::new(MOUNTAIN_CAR_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    /// Batched `MountainCar-v0` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn mountain_car() -> Self {
        Self::new(MOUNTAIN_CAR_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    /// `CartPole-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cart_pole() -> Self {
        Self::new(CART_POLE_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    /// Batched `CartPole-v1` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn cart_pole() -> Self {
        Self::new(CART_POLE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `MountainCarContinuous-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn mountain_car_continuous() -> Self {
        Self::new(MOUNTAIN_CAR_CONTINUOUS_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `MountainCarContinuous-v0` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn mountain_car_continuous() -> Self {
        Self::new(MOUNTAIN_CAR_CONTINUOUS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    /// `Acrobot-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn acrobot() -> Self {
        Self::new(ACROBOT_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    /// Batched `Acrobot-v1` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn acrobot() -> Self {
        Self::new(ACROBOT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Pendulum-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn pendulum() -> Self {
        Self::new(PENDULUM_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Pendulum-v1` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn pendulum() -> Self {
        Self::new(PENDULUM_SPEC)
    }
}

const MOUNTAIN_CAR_SPEC: &'static str = r#"{"id": "MountainCar-v0", "entry_point": "gymnasium.envs.classic_control.mountain_car:MountainCarEnv", "reward_threshold": -110.0, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const CART_POLE_SPEC: &'static str = r#"{"id": "CartPole-v1", "entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleEnv", "reward_threshold": 475.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv"}"#;


const MOUNTAIN_CAR_CONTINUOUS_SPEC: &'static str = r#"{"id": "MountainCarContinuous-v0", "entry_point": "gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv", "reward_threshold": 90.0, "nondeterministic": false, "max_episode_steps": 999, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const ACROBOT_SPEC: &'static str = r#"{"id": "Acrobot-v1", "entry_point": "gymnasium.envs.classic_control.acrobot:AcrobotEnv", "reward_threshold": -100.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const PENDULUM_SPEC: &'static str = r#"{"id": "Pendulum-v1", "entry_point": "gymnasium.envs.classic_control.pendulum:PendulumEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

