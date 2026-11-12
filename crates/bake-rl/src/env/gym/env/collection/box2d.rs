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
    /// `LunarLanderContinuous-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn lunar_lander_continuous() -> Self {
        Self::new(LUNAR_LANDER_CONTINUOUS_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `LunarLanderContinuous-v3` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn lunar_lander_continuous() -> Self {
        Self::new(LUNAR_LANDER_CONTINUOUS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `BipedalWalkerHardcore-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn bipedal_walker_hardcore() -> Self {
        Self::new(BIPEDAL_WALKER_HARDCORE_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `BipedalWalkerHardcore-v3` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn bipedal_walker_hardcore() -> Self {
        Self::new(BIPEDAL_WALKER_HARDCORE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, DiscreteSpace> {
    /// `LunarLander-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn lunar_lander() -> Self {
        Self::new(LUNAR_LANDER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedDiscreteSpace> {
    /// Batched `LunarLander-v3` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn lunar_lander() -> Self {
        Self::new(LUNAR_LANDER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ContinuousSpace<1>, ContinuousSpace<1>> {
    /// `BipedalWalker-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<1>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn bipedal_walker() -> Self {
        Self::new(BIPEDAL_WALKER_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedContinuousSpace<2>, BatchedContinuousSpace<2>> {
    /// Batched `BipedalWalker-v3` environment
    /// - `ObsSpace`: `BatchedContinuousSpace<2>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn bipedal_walker() -> Self {
        Self::new(BIPEDAL_WALKER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, ContinuousSpace<1>> {
    /// `CarRacing-v3` environment
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `ContinuousSpace<1>`
    pub fn car_racing() -> Self {
        Self::new(CAR_RACING_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedByteSpace<4>, BatchedContinuousSpace<2>> {
    /// Batched `CarRacing-v3` environment
    /// - `ObsSpace`: `BatchedByteSpace<4>`
    /// - `ActionSpace`: `BatchedContinuousSpace<2>`
    pub fn car_racing() -> Self {
        Self::new(CAR_RACING_SPEC)
    }
}

const LUNAR_LANDER_CONTINUOUS_SPEC: &'static str = r#"{"id": "LunarLanderContinuous-v3", "entry_point": "gymnasium.envs.box2d.lunar_lander:LunarLander", "reward_threshold": 200, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {"continuous": true}, "additional_wrappers": [], "vector_entry_point": null}"#;


const BIPEDAL_WALKER_HARDCORE_SPEC: &'static str = r#"{"id": "BipedalWalkerHardcore-v3", "entry_point": "gymnasium.envs.box2d.bipedal_walker:BipedalWalker", "reward_threshold": 300, "nondeterministic": false, "max_episode_steps": 2000, "order_enforce": true, "disable_env_checker": false, "kwargs": {"hardcore": true}, "additional_wrappers": [], "vector_entry_point": null}"#;


const LUNAR_LANDER_SPEC: &'static str = r#"{"id": "LunarLander-v3", "entry_point": "gymnasium.envs.box2d.lunar_lander:LunarLander", "reward_threshold": 200, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const BIPEDAL_WALKER_SPEC: &'static str = r#"{"id": "BipedalWalker-v3", "entry_point": "gymnasium.envs.box2d.bipedal_walker:BipedalWalker", "reward_threshold": 300, "nondeterministic": false, "max_episode_steps": 1600, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const CAR_RACING_SPEC: &'static str = r#"{"id": "CarRacing-v3", "entry_point": "gymnasium.envs.box2d.car_racing:CarRacing", "reward_threshold": 900, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

