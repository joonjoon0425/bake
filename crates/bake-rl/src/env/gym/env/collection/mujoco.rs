//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};


/// `Ant-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn ant_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(ANT_SPEC)
}


/// Batched `Ant-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_ant_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(ANT_SPEC)
}


/// `InvertedPendulum-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn inverted_pendulum_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(INVERTED_PENDULUM_SPEC)
}


/// Batched `InvertedPendulum-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_inverted_pendulum_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(INVERTED_PENDULUM_SPEC)
}


/// `Swimmer-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn swimmer_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(SWIMMER_SPEC)
}


/// Batched `Swimmer-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_swimmer_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(SWIMMER_SPEC)
}


/// `Humanoid-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn humanoid_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(HUMANOID_SPEC)
}


/// Batched `Humanoid-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_humanoid_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(HUMANOID_SPEC)
}


/// `Walker2d-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn walker2d_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(WALKER2D_SPEC)
}


/// Batched `Walker2d-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_walker2d_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(WALKER2D_SPEC)
}


/// `HumanoidStandup-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn humanoid_standup_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(HUMANOID_STANDUP_SPEC)
}


/// Batched `HumanoidStandup-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_humanoid_standup_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(HUMANOID_STANDUP_SPEC)
}


/// `Hopper-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn hopper_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(HOPPER_SPEC)
}


/// Batched `Hopper-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_hopper_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(HOPPER_SPEC)
}


/// `Pusher-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn pusher_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(PUSHER_SPEC)
}


/// Batched `Pusher-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_pusher_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(PUSHER_SPEC)
}


/// `InvertedDoublePendulum-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn inverted_double_pendulum_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(INVERTED_DOUBLE_PENDULUM_SPEC)
}


/// Batched `InvertedDoublePendulum-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_inverted_double_pendulum_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(INVERTED_DOUBLE_PENDULUM_SPEC)
}


/// `HalfCheetah-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn half_cheetah_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(HALF_CHEETAH_SPEC)
}


/// Batched `HalfCheetah-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_half_cheetah_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(HALF_CHEETAH_SPEC)
}


/// `Reacher-v5` environment
/// - `ObsSpace`: `ContinuousSpace<1>`
/// - `ActionSpace`: `ContinuousSpace<1>`
pub fn reacher_config() -> GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    GymnasiumEnvironmentConfig::new(REACHER_SPEC)
}


/// Batched `Reacher-v5` environment
/// - `ObsSpace`: `BatchedContinuousSpace<2>`
/// - `ActionSpace`: `BatchedContinuousSpace<2>`
pub fn batched_reacher_config() -> BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    BatchedGymnasiumEnvironmentConfig::new(REACHER_SPEC)
}


const ANT_SPEC: &'static str = r#"{"id": "Ant-v5", "entry_point": "gymnasium.envs.mujoco.ant_v5:AntEnv", "reward_threshold": 6000.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const INVERTED_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedPendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv", "reward_threshold": 950.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const SWIMMER_SPEC: &'static str = r#"{"id": "Swimmer-v5", "entry_point": "gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv", "reward_threshold": 360.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HUMANOID_SPEC: &'static str = r#"{"id": "Humanoid-v5", "entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const WALKER2D_SPEC: &'static str = r#"{"id": "Walker2d-v5", "entry_point": "gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HUMANOID_STANDUP_SPEC: &'static str = r#"{"id": "HumanoidStandup-v5", "entry_point": "gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HOPPER_SPEC: &'static str = r#"{"id": "Hopper-v5", "entry_point": "gymnasium.envs.mujoco.hopper_v5:HopperEnv", "reward_threshold": 3800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const PUSHER_SPEC: &'static str = r#"{"id": "Pusher-v5", "entry_point": "gymnasium.envs.mujoco.pusher_v5:PusherEnv", "reward_threshold": 0.0, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const INVERTED_DOUBLE_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedDoublePendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv", "reward_threshold": 9100.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HALF_CHEETAH_SPEC: &'static str = r#"{"id": "HalfCheetah-v5", "entry_point": "gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv", "reward_threshold": 4800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const REACHER_SPEC: &'static str = r#"{"id": "Reacher-v5", "entry_point": "gymnasium.envs.mujoco.reacher_v5:ReacherEnv", "reward_threshold": -3.75, "nondeterministic": false, "max_episode_steps": 50, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

