//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `InvertedPendulum-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn inverted_pendulum() -> Self {
        Self::new(INVERTED_PENDULUM_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `InvertedPendulum-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn inverted_pendulum() -> Self {
        Self::new(INVERTED_PENDULUM_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `HalfCheetah-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn half_cheetah() -> Self {
        Self::new(HALF_CHEETAH_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `HalfCheetah-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn half_cheetah() -> Self {
        Self::new(HALF_CHEETAH_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `HumanoidStandup-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn humanoid_standup() -> Self {
        Self::new(HUMANOID_STANDUP_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `HumanoidStandup-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn humanoid_standup() -> Self {
        Self::new(HUMANOID_STANDUP_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Swimmer-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn swimmer() -> Self {
        Self::new(SWIMMER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Swimmer-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn swimmer() -> Self {
        Self::new(SWIMMER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Walker2d-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn walker2d() -> Self {
        Self::new(WALKER2D_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Walker2d-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn walker2d() -> Self {
        Self::new(WALKER2D_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Ant-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn ant() -> Self {
        Self::new(ANT_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Ant-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn ant() -> Self {
        Self::new(ANT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Humanoid-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn humanoid() -> Self {
        Self::new(HUMANOID_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Humanoid-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn humanoid() -> Self {
        Self::new(HUMANOID_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `InvertedDoublePendulum-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn inverted_double_pendulum() -> Self {
        Self::new(INVERTED_DOUBLE_PENDULUM_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `InvertedDoublePendulum-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn inverted_double_pendulum() -> Self {
        Self::new(INVERTED_DOUBLE_PENDULUM_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Pusher-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn pusher() -> Self {
        Self::new(PUSHER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Pusher-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn pusher() -> Self {
        Self::new(PUSHER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Reacher-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn reacher() -> Self {
        Self::new(REACHER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Reacher-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn reacher() -> Self {
        Self::new(REACHER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `Hopper-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn hopper() -> Self {
        Self::new(HOPPER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `Hopper-v5` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn hopper() -> Self {
        Self::new(HOPPER_SPEC)
    }
}

const INVERTED_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedPendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv", "reward_threshold": 950.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HALF_CHEETAH_SPEC: &'static str = r#"{"id": "HalfCheetah-v5", "entry_point": "gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv", "reward_threshold": 4800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HUMANOID_STANDUP_SPEC: &'static str = r#"{"id": "HumanoidStandup-v5", "entry_point": "gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const SWIMMER_SPEC: &'static str = r#"{"id": "Swimmer-v5", "entry_point": "gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv", "reward_threshold": 360.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const WALKER2D_SPEC: &'static str = r#"{"id": "Walker2d-v5", "entry_point": "gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const ANT_SPEC: &'static str = r#"{"id": "Ant-v5", "entry_point": "gymnasium.envs.mujoco.ant_v5:AntEnv", "reward_threshold": 6000.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HUMANOID_SPEC: &'static str = r#"{"id": "Humanoid-v5", "entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const INVERTED_DOUBLE_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedDoublePendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv", "reward_threshold": 9100.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const PUSHER_SPEC: &'static str = r#"{"id": "Pusher-v5", "entry_point": "gymnasium.envs.mujoco.pusher_v5:PusherEnv", "reward_threshold": 0.0, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const REACHER_SPEC: &'static str = r#"{"id": "Reacher-v5", "entry_point": "gymnasium.envs.mujoco.reacher_v5:ReacherEnv", "reward_threshold": -3.75, "nondeterministic": false, "max_episode_steps": 50, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const HOPPER_SPEC: &'static str = r#"{"id": "Hopper-v5", "entry_point": "gymnasium.envs.mujoco.hopper_v5:HopperEnv", "reward_threshold": 3800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

