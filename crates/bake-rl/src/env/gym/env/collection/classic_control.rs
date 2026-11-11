//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};


/// `CartPole-v1` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `DiscreteSpace`
pub fn cart_pole_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new(CART_POLE_SPEC)
}


/// Batched `CartPole-v1` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedDiscreteSpace`
pub fn batched_cart_pole_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    BatchedGymnasiumEnvironmentConfig::new(CART_POLE_SPEC)
}


/// `MountainCar-v0` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `DiscreteSpace`
pub fn mountain_car_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new(MOUNTAIN_CAR_SPEC)
}


/// Batched `MountainCar-v0` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedDiscreteSpace`
pub fn batched_mountain_car_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    BatchedGymnasiumEnvironmentConfig::new(MOUNTAIN_CAR_SPEC)
}


/// `Acrobot-v1` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `DiscreteSpace`
pub fn acrobot_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    GymnasiumEnvironmentConfig::new(ACROBOT_SPEC)
}


/// Batched `Acrobot-v1` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedDiscreteSpace`
pub fn batched_acrobot_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    BatchedGymnasiumEnvironmentConfig::new(ACROBOT_SPEC)
}


/// `Pendulum-v1` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn pendulum_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(PENDULUM_SPEC)
}


/// Batched `Pendulum-v1` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_pendulum_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(PENDULUM_SPEC)
}


/// `MountainCarContinuous-v0` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn mountain_car_continuous_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(MOUNTAIN_CAR_CONTINUOUS_SPEC)
}


/// Batched `MountainCarContinuous-v0` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_mountain_car_continuous_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(MOUNTAIN_CAR_CONTINUOUS_SPEC)
}


const CART_POLE_SPEC: &'static str = r#"{"id": "CartPole-v1", "entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleEnv", "reward_threshold": 475.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv"}"#;


const MOUNTAIN_CAR_SPEC: &'static str = r#"{"id": "MountainCar-v0", "entry_point": "gymnasium.envs.classic_control.mountain_car:MountainCarEnv", "reward_threshold": -110.0, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const ACROBOT_SPEC: &'static str = r#"{"id": "Acrobot-v1", "entry_point": "gymnasium.envs.classic_control.acrobot:AcrobotEnv", "reward_threshold": -100.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const PENDULUM_SPEC: &'static str = r#"{"id": "Pendulum-v1", "entry_point": "gymnasium.envs.classic_control.pendulum:PendulumEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const MOUNTAIN_CAR_CONTINUOUS_SPEC: &'static str = r#"{"id": "MountainCarContinuous-v0", "entry_point": "gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv", "reward_threshold": 90.0, "nondeterministic": false, "max_episode_steps": 999, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

