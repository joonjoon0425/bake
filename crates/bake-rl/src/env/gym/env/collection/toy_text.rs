//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `CliffWalking-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cliff_walking() -> Self {
        Self::new(CLIFF_WALKING_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedDiscreteSpace, BatchedDiscreteSpace> {
    /// Batched `CliffWalking-v1` environment
    /// - `ObsSpace`: `BatchedDiscreteSpace`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn cliff_walking() -> Self {
        Self::new(CLIFF_WALKING_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `CliffWalkingSlippery-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cliff_walking_slippery() -> Self {
        Self::new(CLIFF_WALKING_SLIPPERY_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedDiscreteSpace, BatchedDiscreteSpace> {
    /// Batched `CliffWalkingSlippery-v1` environment
    /// - `ObsSpace`: `BatchedDiscreteSpace`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn cliff_walking_slippery() -> Self {
        Self::new(CLIFF_WALKING_SLIPPERY_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `FrozenLake8x8-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frozen_lake8x8() -> Self {
        Self::new(FROZEN_LAKE8X8_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedDiscreteSpace, BatchedDiscreteSpace> {
    /// Batched `FrozenLake8x8-v1` environment
    /// - `ObsSpace`: `BatchedDiscreteSpace`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn frozen_lake8x8() -> Self {
        Self::new(FROZEN_LAKE8X8_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `FrozenLake-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frozen_lake() -> Self {
        Self::new(FROZEN_LAKE_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedDiscreteSpace, BatchedDiscreteSpace> {
    /// Batched `FrozenLake-v1` environment
    /// - `ObsSpace`: `BatchedDiscreteSpace`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn frozen_lake() -> Self {
        Self::new(FROZEN_LAKE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `Taxi-v4` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn taxi() -> Self {
        Self::new(TAXI_SPEC)
    }
}
impl BatchedGymnasiumEnvironmentConfig<BatchedDiscreteSpace, BatchedDiscreteSpace> {
    /// Batched `Taxi-v4` environment
    /// - `ObsSpace`: `BatchedDiscreteSpace`
    /// - `ActionSpace`: `BatchedDiscreteSpace`
    pub fn taxi() -> Self {
        Self::new(TAXI_SPEC)
    }
}

// blackjack not implemented due to following reason: ("bake do not have `Space` correspoding to Gymnasium space of type <class 'gymnasium.spaces.tuple.Tuple'> currently",)


const CLIFF_WALKING_SPEC: &'static str = r#"{"id": "CliffWalking-v1", "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


const CLIFF_WALKING_SLIPPERY_SPEC: &'static str = r#"{"id": "CliffWalkingSlippery-v1", "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"is_slippery": true}, "additional_wrappers": [], "vector_entry_point": null}"#;


const FROZEN_LAKE8X8_SPEC: &'static str = r#"{"id": "FrozenLake8x8-v1", "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv", "reward_threshold": 0.85, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {"map_name": "8x8"}, "additional_wrappers": [], "vector_entry_point": null}"#;


const FROZEN_LAKE_SPEC: &'static str = r#"{"id": "FrozenLake-v1", "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv", "reward_threshold": 0.7, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {"map_name": "4x4"}, "additional_wrappers": [], "vector_entry_point": null}"#;


const TAXI_SPEC: &'static str = r#"{"id": "Taxi-v4", "entry_point": "gymnasium.envs.toy_text.taxi:TaxiEnv", "reward_threshold": 8, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

