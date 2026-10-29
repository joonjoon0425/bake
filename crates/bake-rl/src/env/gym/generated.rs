//! DO NOT EDIT
//! Auto-generated code with script/gym_gen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/gym_gen.py`

use crate::env::gym::env::GymnasiumEnvironmentConfig;
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `CartPole-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cart_pole() -> Self {
        Self::new(CART_POLE_SPEC)
    }
}
const CART_POLE_SPEC: &'static str = r#"{"id": "CartPole-v1", "entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleEnv", "reward_threshold": 475.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `MountainCar-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mountain_car() -> Self {
        Self::new(MOUNTAIN_CAR_SPEC)
    }
}
const MOUNTAIN_CAR_SPEC: &'static str = r#"{"id": "MountainCar-v0", "entry_point": "gymnasium.envs.classic_control.mountain_car:MountainCarEnv", "reward_threshold": -110.0, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `MountainCarContinuous-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn mountain_car_continuous() -> Self {
        Self::new(MOUNTAIN_CAR_CONTINUOUS_SPEC)
    }
}
const MOUNTAIN_CAR_CONTINUOUS_SPEC: &'static str = r#"{"id": "MountainCarContinuous-v0", "entry_point": "gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv", "reward_threshold": 90.0, "nondeterministic": false, "max_episode_steps": 999, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Pendulum-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn pendulum() -> Self {
        Self::new(PENDULUM_SPEC)
    }
}
const PENDULUM_SPEC: &'static str = r#"{"id": "Pendulum-v1", "entry_point": "gymnasium.envs.classic_control.pendulum:PendulumEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `Acrobot-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn acrobot() -> Self {
        Self::new(ACROBOT_SPEC)
    }
}
const ACROBOT_SPEC: &'static str = r#"{"id": "Acrobot-v1", "entry_point": "gymnasium.envs.classic_control.acrobot:AcrobotEnv", "reward_threshold": -100.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `LunarLander-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn lunar_lander() -> Self {
        Self::new(LUNAR_LANDER_SPEC)
    }
}
const LUNAR_LANDER_SPEC: &'static str = r#"{"id": "LunarLander-v3", "entry_point": "gymnasium.envs.box2d.lunar_lander:LunarLander", "reward_threshold": 200, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `LunarLanderContinuous-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn lunar_lander_continuous() -> Self {
        Self::new(LUNAR_LANDER_CONTINUOUS_SPEC)
    }
}
const LUNAR_LANDER_CONTINUOUS_SPEC: &'static str = r#"{"id": "LunarLanderContinuous-v3", "entry_point": "gymnasium.envs.box2d.lunar_lander:LunarLander", "reward_threshold": 200, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {"continuous": true}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `BipedalWalker-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn bipedal_walker() -> Self {
        Self::new(BIPEDAL_WALKER_SPEC)
    }
}
const BIPEDAL_WALKER_SPEC: &'static str = r#"{"id": "BipedalWalker-v3", "entry_point": "gymnasium.envs.box2d.bipedal_walker:BipedalWalker", "reward_threshold": 300, "nondeterministic": false, "max_episode_steps": 1600, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `BipedalWalkerHardcore-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn bipedal_walker_hardcore() -> Self {
        Self::new(BIPEDAL_WALKER_HARDCORE_SPEC)
    }
}
const BIPEDAL_WALKER_HARDCORE_SPEC: &'static str = r#"{"id": "BipedalWalkerHardcore-v3", "entry_point": "gymnasium.envs.box2d.bipedal_walker:BipedalWalker", "reward_threshold": 300, "nondeterministic": false, "max_episode_steps": 2000, "order_enforce": true, "disable_env_checker": false, "kwargs": {"hardcore": true}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, ContinuousSpace<2>> {
    /// `CarRacing-v3` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn car_racing() -> Self {
        Self::new(CAR_RACING_SPEC)
    }
}
const CAR_RACING_SPEC: &'static str = r#"{"id": "CarRacing-v3", "entry_point": "gymnasium.envs.box2d.car_racing:CarRacing", "reward_threshold": 900, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;


// blackjack not implemented due to following reason: ("bake do not have `Space` correspoding to Gymnasium space of type <class 'gymnasium.spaces.tuple.Tuple'> currently",)

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `FrozenLake-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frozen_lake() -> Self {
        Self::new(FROZEN_LAKE_SPEC)
    }
}
const FROZEN_LAKE_SPEC: &'static str = r#"{"id": "FrozenLake-v1", "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv", "reward_threshold": 0.7, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {"map_name": "4x4"}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `FrozenLake8x8-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frozen_lake8x8() -> Self {
        Self::new(FROZEN_LAKE8X8_SPEC)
    }
}
const FROZEN_LAKE8X8_SPEC: &'static str = r#"{"id": "FrozenLake8x8-v1", "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv", "reward_threshold": 0.85, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {"map_name": "8x8"}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `CliffWalking-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cliff_walking() -> Self {
        Self::new(CLIFF_WALKING_SPEC)
    }
}
const CLIFF_WALKING_SPEC: &'static str = r#"{"id": "CliffWalking-v1", "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `CliffWalkingSlippery-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cliff_walking_slippery() -> Self {
        Self::new(CLIFF_WALKING_SLIPPERY_SPEC)
    }
}
const CLIFF_WALKING_SLIPPERY_SPEC: &'static str = r#"{"id": "CliffWalkingSlippery-v1", "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"is_slippery": true}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `Taxi-v4` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn taxi() -> Self {
        Self::new(TAXI_SPEC)
    }
}
const TAXI_SPEC: &'static str = r#"{"id": "Taxi-v4", "entry_point": "gymnasium.envs.toy_text.taxi:TaxiEnv", "reward_threshold": 8, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Reacher-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn reacher() -> Self {
        Self::new(REACHER_SPEC)
    }
}
const REACHER_SPEC: &'static str = r#"{"id": "Reacher-v5", "entry_point": "gymnasium.envs.mujoco.reacher_v5:ReacherEnv", "reward_threshold": -3.75, "nondeterministic": false, "max_episode_steps": 50, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Pusher-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn pusher() -> Self {
        Self::new(PUSHER_SPEC)
    }
}
const PUSHER_SPEC: &'static str = r#"{"id": "Pusher-v5", "entry_point": "gymnasium.envs.mujoco.pusher_v5:PusherEnv", "reward_threshold": 0.0, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `InvertedPendulum-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn inverted_pendulum() -> Self {
        Self::new(INVERTED_PENDULUM_SPEC)
    }
}
const INVERTED_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedPendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv", "reward_threshold": 950.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `InvertedDoublePendulum-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn inverted_double_pendulum() -> Self {
        Self::new(INVERTED_DOUBLE_PENDULUM_SPEC)
    }
}
const INVERTED_DOUBLE_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedDoublePendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv", "reward_threshold": 9100.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `HalfCheetah-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn half_cheetah() -> Self {
        Self::new(HALF_CHEETAH_SPEC)
    }
}
const HALF_CHEETAH_SPEC: &'static str = r#"{"id": "HalfCheetah-v5", "entry_point": "gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv", "reward_threshold": 4800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Hopper-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn hopper() -> Self {
        Self::new(HOPPER_SPEC)
    }
}
const HOPPER_SPEC: &'static str = r#"{"id": "Hopper-v5", "entry_point": "gymnasium.envs.mujoco.hopper_v5:HopperEnv", "reward_threshold": 3800.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Swimmer-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn swimmer() -> Self {
        Self::new(SWIMMER_SPEC)
    }
}
const SWIMMER_SPEC: &'static str = r#"{"id": "Swimmer-v5", "entry_point": "gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv", "reward_threshold": 360.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Walker2d-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn walker2d() -> Self {
        Self::new(WALKER2D_SPEC)
    }
}
const WALKER2D_SPEC: &'static str = r#"{"id": "Walker2d-v5", "entry_point": "gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Ant-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn ant() -> Self {
        Self::new(ANT_SPEC)
    }
}
const ANT_SPEC: &'static str = r#"{"id": "Ant-v5", "entry_point": "gymnasium.envs.mujoco.ant_v5:AntEnv", "reward_threshold": 6000.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `Humanoid-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn humanoid() -> Self {
        Self::new(HUMANOID_SPEC)
    }
}
const HUMANOID_SPEC: &'static str = r#"{"id": "Humanoid-v5", "entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `HumanoidStandup-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn humanoid_standup() -> Self {
        Self::new(HUMANOID_STANDUP_SPEC)
    }
}
const HUMANOID_STANDUP_SPEC: &'static str = r#"{"id": "HumanoidStandup-v5", "entry_point": "gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Adventure-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn adventure_preprocessed() -> Self {
        Self::new(ADVENTURE_PREPROCESSED_SPEC)
    }
}
const ADVENTURE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Adventure-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "adventure", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/AirRaid-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn air_raid_preprocessed() -> Self {
        Self::new(AIR_RAID_PREPROCESSED_SPEC)
    }
}
const AIR_RAID_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/AirRaid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "air_raid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Alien-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn alien_preprocessed() -> Self {
        Self::new(ALIEN_PREPROCESSED_SPEC)
    }
}
const ALIEN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Alien-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "alien", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Amidar-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn amidar_preprocessed() -> Self {
        Self::new(AMIDAR_PREPROCESSED_SPEC)
    }
}
const AMIDAR_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Amidar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "amidar", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Assault-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn assault_preprocessed() -> Self {
        Self::new(ASSAULT_PREPROCESSED_SPEC)
    }
}
const ASSAULT_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Assault-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "assault", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Asterix-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn asterix_preprocessed() -> Self {
        Self::new(ASTERIX_PREPROCESSED_SPEC)
    }
}
const ASTERIX_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Asterix-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "asterix", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Asteroids-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn asteroids_preprocessed() -> Self {
        Self::new(ASTEROIDS_PREPROCESSED_SPEC)
    }
}
const ASTEROIDS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Asteroids-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "asteroids", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Atlantis2-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn atlantis2_preprocessed() -> Self {
        Self::new(ATLANTIS2_PREPROCESSED_SPEC)
    }
}
const ATLANTIS2_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Atlantis2-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "atlantis2", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Atlantis-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn atlantis_preprocessed() -> Self {
        Self::new(ATLANTIS_PREPROCESSED_SPEC)
    }
}
const ATLANTIS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Atlantis-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "atlantis", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/BankHeist-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn bank_heist_preprocessed() -> Self {
        Self::new(BANK_HEIST_PREPROCESSED_SPEC)
    }
}
const BANK_HEIST_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/BankHeist-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "bank_heist", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/BasicMath-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn basic_math_preprocessed() -> Self {
        Self::new(BASIC_MATH_PREPROCESSED_SPEC)
    }
}
const BASIC_MATH_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/BasicMath-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "basic_math", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/BattleZone-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn battle_zone_preprocessed() -> Self {
        Self::new(BATTLE_ZONE_PREPROCESSED_SPEC)
    }
}
const BATTLE_ZONE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/BattleZone-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "battle_zone", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/BeamRider-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn beam_rider_preprocessed() -> Self {
        Self::new(BEAM_RIDER_PREPROCESSED_SPEC)
    }
}
const BEAM_RIDER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/BeamRider-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "beam_rider", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Berzerk-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn berzerk_preprocessed() -> Self {
        Self::new(BERZERK_PREPROCESSED_SPEC)
    }
}
const BERZERK_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Berzerk-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "berzerk", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Blackjack-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn blackjack_preprocessed() -> Self {
        Self::new(BLACKJACK_PREPROCESSED_SPEC)
    }
}
const BLACKJACK_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Blackjack-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "blackjack", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Bowling-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn bowling_preprocessed() -> Self {
        Self::new(BOWLING_PREPROCESSED_SPEC)
    }
}
const BOWLING_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Bowling-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "bowling", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Boxing-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn boxing_preprocessed() -> Self {
        Self::new(BOXING_PREPROCESSED_SPEC)
    }
}
const BOXING_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Boxing-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "boxing", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Breakout-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn breakout_preprocessed() -> Self {
        Self::new(BREAKOUT_PREPROCESSED_SPEC)
    }
}
const BREAKOUT_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Breakout-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "breakout", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Carnival-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn carnival_preprocessed() -> Self {
        Self::new(CARNIVAL_PREPROCESSED_SPEC)
    }
}
const CARNIVAL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Carnival-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "carnival", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Casino-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn casino_preprocessed() -> Self {
        Self::new(CASINO_PREPROCESSED_SPEC)
    }
}
const CASINO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Casino-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "casino", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Centipede-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn centipede_preprocessed() -> Self {
        Self::new(CENTIPEDE_PREPROCESSED_SPEC)
    }
}
const CENTIPEDE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Centipede-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "centipede", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/ChopperCommand-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn chopper_command_preprocessed() -> Self {
        Self::new(CHOPPER_COMMAND_PREPROCESSED_SPEC)
    }
}
const CHOPPER_COMMAND_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/ChopperCommand-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "chopper_command", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/CrazyClimber-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn crazy_climber_preprocessed() -> Self {
        Self::new(CRAZY_CLIMBER_PREPROCESSED_SPEC)
    }
}
const CRAZY_CLIMBER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/CrazyClimber-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "crazy_climber", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Crossbow-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn crossbow_preprocessed() -> Self {
        Self::new(CROSSBOW_PREPROCESSED_SPEC)
    }
}
const CROSSBOW_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Crossbow-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "crossbow", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Darkchambers-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn darkchambers_preprocessed() -> Self {
        Self::new(DARKCHAMBERS_PREPROCESSED_SPEC)
    }
}
const DARKCHAMBERS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Darkchambers-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "darkchambers", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Defender-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn defender_preprocessed() -> Self {
        Self::new(DEFENDER_PREPROCESSED_SPEC)
    }
}
const DEFENDER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Defender-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "defender", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/DemonAttack-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn demon_attack_preprocessed() -> Self {
        Self::new(DEMON_ATTACK_PREPROCESSED_SPEC)
    }
}
const DEMON_ATTACK_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/DemonAttack-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "demon_attack", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/DonkeyKong-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn donkey_kong_preprocessed() -> Self {
        Self::new(DONKEY_KONG_PREPROCESSED_SPEC)
    }
}
const DONKEY_KONG_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/DonkeyKong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "donkey_kong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/DoubleDunk-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn double_dunk_preprocessed() -> Self {
        Self::new(DOUBLE_DUNK_PREPROCESSED_SPEC)
    }
}
const DOUBLE_DUNK_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/DoubleDunk-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "double_dunk", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Earthworld-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn earthworld_preprocessed() -> Self {
        Self::new(EARTHWORLD_PREPROCESSED_SPEC)
    }
}
const EARTHWORLD_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Earthworld-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "earthworld", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/ElevatorAction-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn elevator_action_preprocessed() -> Self {
        Self::new(ELEVATOR_ACTION_PREPROCESSED_SPEC)
    }
}
const ELEVATOR_ACTION_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/ElevatorAction-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "elevator_action", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Enduro-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn enduro_preprocessed() -> Self {
        Self::new(ENDURO_PREPROCESSED_SPEC)
    }
}
const ENDURO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Enduro-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "enduro", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Entombed-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn entombed_preprocessed() -> Self {
        Self::new(ENTOMBED_PREPROCESSED_SPEC)
    }
}
const ENTOMBED_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Entombed-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "entombed", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Et-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn et_preprocessed() -> Self {
        Self::new(ET_PREPROCESSED_SPEC)
    }
}
const ET_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Et-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "et", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/FishingDerby-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn fishing_derby_preprocessed() -> Self {
        Self::new(FISHING_DERBY_PREPROCESSED_SPEC)
    }
}
const FISHING_DERBY_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/FishingDerby-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "fishing_derby", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/FlagCapture-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn flag_capture_preprocessed() -> Self {
        Self::new(FLAG_CAPTURE_PREPROCESSED_SPEC)
    }
}
const FLAG_CAPTURE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/FlagCapture-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "flag_capture", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Freeway-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn freeway_preprocessed() -> Self {
        Self::new(FREEWAY_PREPROCESSED_SPEC)
    }
}
const FREEWAY_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Freeway-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "freeway", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Frogger-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frogger_preprocessed() -> Self {
        Self::new(FROGGER_PREPROCESSED_SPEC)
    }
}
const FROGGER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Frogger-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "frogger", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Frostbite-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frostbite_preprocessed() -> Self {
        Self::new(FROSTBITE_PREPROCESSED_SPEC)
    }
}
const FROSTBITE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Frostbite-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "frostbite", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Galaxian-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn galaxian_preprocessed() -> Self {
        Self::new(GALAXIAN_PREPROCESSED_SPEC)
    }
}
const GALAXIAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Galaxian-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "galaxian", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Gopher-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn gopher_preprocessed() -> Self {
        Self::new(GOPHER_PREPROCESSED_SPEC)
    }
}
const GOPHER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Gopher-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "gopher", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Gravitar-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn gravitar_preprocessed() -> Self {
        Self::new(GRAVITAR_PREPROCESSED_SPEC)
    }
}
const GRAVITAR_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Gravitar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "gravitar", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Hangman-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn hangman_preprocessed() -> Self {
        Self::new(HANGMAN_PREPROCESSED_SPEC)
    }
}
const HANGMAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Hangman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "hangman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/HauntedHouse-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn haunted_house_preprocessed() -> Self {
        Self::new(HAUNTED_HOUSE_PREPROCESSED_SPEC)
    }
}
const HAUNTED_HOUSE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/HauntedHouse-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "haunted_house", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Hero-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn hero_preprocessed() -> Self {
        Self::new(HERO_PREPROCESSED_SPEC)
    }
}
const HERO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Hero-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "hero", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/HumanCannonball-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn human_cannonball_preprocessed() -> Self {
        Self::new(HUMAN_CANNONBALL_PREPROCESSED_SPEC)
    }
}
const HUMAN_CANNONBALL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/HumanCannonball-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "human_cannonball", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/IceHockey-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn ice_hockey_preprocessed() -> Self {
        Self::new(ICE_HOCKEY_PREPROCESSED_SPEC)
    }
}
const ICE_HOCKEY_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/IceHockey-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "ice_hockey", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Jamesbond-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn jamesbond_preprocessed() -> Self {
        Self::new(JAMESBOND_PREPROCESSED_SPEC)
    }
}
const JAMESBOND_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Jamesbond-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "jamesbond", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/JourneyEscape-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn journey_escape_preprocessed() -> Self {
        Self::new(JOURNEY_ESCAPE_PREPROCESSED_SPEC)
    }
}
const JOURNEY_ESCAPE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/JourneyEscape-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "journey_escape", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Kaboom-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kaboom_preprocessed() -> Self {
        Self::new(KABOOM_PREPROCESSED_SPEC)
    }
}
const KABOOM_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Kaboom-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kaboom", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Kangaroo-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kangaroo_preprocessed() -> Self {
        Self::new(KANGAROO_PREPROCESSED_SPEC)
    }
}
const KANGAROO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Kangaroo-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kangaroo", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/KeystoneKapers-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn keystone_kapers_preprocessed() -> Self {
        Self::new(KEYSTONE_KAPERS_PREPROCESSED_SPEC)
    }
}
const KEYSTONE_KAPERS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/KeystoneKapers-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "keystone_kapers", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/KingKong-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn king_kong_preprocessed() -> Self {
        Self::new(KING_KONG_PREPROCESSED_SPEC)
    }
}
const KING_KONG_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/KingKong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "king_kong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Klax-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn klax_preprocessed() -> Self {
        Self::new(KLAX_PREPROCESSED_SPEC)
    }
}
const KLAX_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Klax-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "klax", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Koolaid-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn koolaid_preprocessed() -> Self {
        Self::new(KOOLAID_PREPROCESSED_SPEC)
    }
}
const KOOLAID_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Koolaid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "koolaid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Krull-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn krull_preprocessed() -> Self {
        Self::new(KRULL_PREPROCESSED_SPEC)
    }
}
const KRULL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Krull-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "krull", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/KungFuMaster-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kung_fu_master_preprocessed() -> Self {
        Self::new(KUNG_FU_MASTER_PREPROCESSED_SPEC)
    }
}
const KUNG_FU_MASTER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/KungFuMaster-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kung_fu_master", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/LaserGates-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn laser_gates_preprocessed() -> Self {
        Self::new(LASER_GATES_PREPROCESSED_SPEC)
    }
}
const LASER_GATES_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/LaserGates-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "laser_gates", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/LostLuggage-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn lost_luggage_preprocessed() -> Self {
        Self::new(LOST_LUGGAGE_PREPROCESSED_SPEC)
    }
}
const LOST_LUGGAGE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/LostLuggage-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "lost_luggage", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/MarioBros-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mario_bros_preprocessed() -> Self {
        Self::new(MARIO_BROS_PREPROCESSED_SPEC)
    }
}
const MARIO_BROS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/MarioBros-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "mario_bros", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/MiniatureGolf-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn miniature_golf_preprocessed() -> Self {
        Self::new(MINIATURE_GOLF_PREPROCESSED_SPEC)
    }
}
const MINIATURE_GOLF_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/MiniatureGolf-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "miniature_golf", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/MontezumaRevenge-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn montezuma_revenge_preprocessed() -> Self {
        Self::new(MONTEZUMA_REVENGE_PREPROCESSED_SPEC)
    }
}
const MONTEZUMA_REVENGE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/MontezumaRevenge-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "montezuma_revenge", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/MrDo-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mr_do_preprocessed() -> Self {
        Self::new(MR_DO_PREPROCESSED_SPEC)
    }
}
const MR_DO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/MrDo-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "mr_do", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/MsPacman-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn ms_pacman_preprocessed() -> Self {
        Self::new(MS_PACMAN_PREPROCESSED_SPEC)
    }
}
const MS_PACMAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/MsPacman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "ms_pacman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/NameThisGame-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn name_this_game_preprocessed() -> Self {
        Self::new(NAME_THIS_GAME_PREPROCESSED_SPEC)
    }
}
const NAME_THIS_GAME_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/NameThisGame-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "name_this_game", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Othello-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn othello_preprocessed() -> Self {
        Self::new(OTHELLO_PREPROCESSED_SPEC)
    }
}
const OTHELLO_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Othello-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "othello", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pacman-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pacman_preprocessed() -> Self {
        Self::new(PACMAN_PREPROCESSED_SPEC)
    }
}
const PACMAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pacman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pacman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Phoenix-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn phoenix_preprocessed() -> Self {
        Self::new(PHOENIX_PREPROCESSED_SPEC)
    }
}
const PHOENIX_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Phoenix-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "phoenix", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pitfall2-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pitfall2_preprocessed() -> Self {
        Self::new(PITFALL2_PREPROCESSED_SPEC)
    }
}
const PITFALL2_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pitfall2-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pitfall2", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pitfall-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pitfall_preprocessed() -> Self {
        Self::new(PITFALL_PREPROCESSED_SPEC)
    }
}
const PITFALL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pitfall-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pitfall", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pong-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pong_preprocessed() -> Self {
        Self::new(PONG_PREPROCESSED_SPEC)
    }
}
const PONG_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pooyan-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pooyan_preprocessed() -> Self {
        Self::new(POOYAN_PREPROCESSED_SPEC)
    }
}
const POOYAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pooyan-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pooyan", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/PrivateEye-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn private_eye_preprocessed() -> Self {
        Self::new(PRIVATE_EYE_PREPROCESSED_SPEC)
    }
}
const PRIVATE_EYE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/PrivateEye-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "private_eye", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Qbert-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn qbert_preprocessed() -> Self {
        Self::new(QBERT_PREPROCESSED_SPEC)
    }
}
const QBERT_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Qbert-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "qbert", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Riverraid-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn riverraid_preprocessed() -> Self {
        Self::new(RIVERRAID_PREPROCESSED_SPEC)
    }
}
const RIVERRAID_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Riverraid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "riverraid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/RoadRunner-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn road_runner_preprocessed() -> Self {
        Self::new(ROAD_RUNNER_PREPROCESSED_SPEC)
    }
}
const ROAD_RUNNER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/RoadRunner-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "road_runner", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Robotank-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn robotank_preprocessed() -> Self {
        Self::new(ROBOTANK_PREPROCESSED_SPEC)
    }
}
const ROBOTANK_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Robotank-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "robotank", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Seaquest-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn seaquest_preprocessed() -> Self {
        Self::new(SEAQUEST_PREPROCESSED_SPEC)
    }
}
const SEAQUEST_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Seaquest-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "seaquest", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/SirLancelot-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn sir_lancelot_preprocessed() -> Self {
        Self::new(SIR_LANCELOT_PREPROCESSED_SPEC)
    }
}
const SIR_LANCELOT_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/SirLancelot-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "sir_lancelot", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Skiing-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn skiing_preprocessed() -> Self {
        Self::new(SKIING_PREPROCESSED_SPEC)
    }
}
const SKIING_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Skiing-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "skiing", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Solaris-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn solaris_preprocessed() -> Self {
        Self::new(SOLARIS_PREPROCESSED_SPEC)
    }
}
const SOLARIS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Solaris-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "solaris", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/SpaceInvaders-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn space_invaders_preprocessed() -> Self {
        Self::new(SPACE_INVADERS_PREPROCESSED_SPEC)
    }
}
const SPACE_INVADERS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/SpaceInvaders-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "space_invaders", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/SpaceWar-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn space_war_preprocessed() -> Self {
        Self::new(SPACE_WAR_PREPROCESSED_SPEC)
    }
}
const SPACE_WAR_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/SpaceWar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "space_war", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/StarGunner-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn star_gunner_preprocessed() -> Self {
        Self::new(STAR_GUNNER_PREPROCESSED_SPEC)
    }
}
const STAR_GUNNER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/StarGunner-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "star_gunner", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Superman-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn superman_preprocessed() -> Self {
        Self::new(SUPERMAN_PREPROCESSED_SPEC)
    }
}
const SUPERMAN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Superman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "superman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Surround-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn surround_preprocessed() -> Self {
        Self::new(SURROUND_PREPROCESSED_SPEC)
    }
}
const SURROUND_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Surround-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "surround", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Tennis-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tennis_preprocessed() -> Self {
        Self::new(TENNIS_PREPROCESSED_SPEC)
    }
}
const TENNIS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Tennis-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tennis", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Tetris-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tetris_preprocessed() -> Self {
        Self::new(TETRIS_PREPROCESSED_SPEC)
    }
}
const TETRIS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Tetris-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tetris", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/TicTacToe3D-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tic_tac_toe3d_preprocessed() -> Self {
        Self::new(TIC_TAC_TOE3D_PREPROCESSED_SPEC)
    }
}
const TIC_TAC_TOE3D_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/TicTacToe3D-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tic_tac_toe_3d", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/TimePilot-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn time_pilot_preprocessed() -> Self {
        Self::new(TIME_PILOT_PREPROCESSED_SPEC)
    }
}
const TIME_PILOT_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/TimePilot-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "time_pilot", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Trondead-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn trondead_preprocessed() -> Self {
        Self::new(TRONDEAD_PREPROCESSED_SPEC)
    }
}
const TRONDEAD_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Trondead-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "trondead", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Turmoil-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn turmoil_preprocessed() -> Self {
        Self::new(TURMOIL_PREPROCESSED_SPEC)
    }
}
const TURMOIL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Turmoil-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "turmoil", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Tutankham-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tutankham_preprocessed() -> Self {
        Self::new(TUTANKHAM_PREPROCESSED_SPEC)
    }
}
const TUTANKHAM_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Tutankham-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tutankham", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/UpNDown-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn up_n_down_preprocessed() -> Self {
        Self::new(UP_N_DOWN_PREPROCESSED_SPEC)
    }
}
const UP_N_DOWN_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/UpNDown-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "up_n_down", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Venture-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn venture_preprocessed() -> Self {
        Self::new(VENTURE_PREPROCESSED_SPEC)
    }
}
const VENTURE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Venture-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "venture", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/VideoChess-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_chess_preprocessed() -> Self {
        Self::new(VIDEO_CHESS_PREPROCESSED_SPEC)
    }
}
const VIDEO_CHESS_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/VideoChess-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_chess", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/VideoCube-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_cube_preprocessed() -> Self {
        Self::new(VIDEO_CUBE_PREPROCESSED_SPEC)
    }
}
const VIDEO_CUBE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/VideoCube-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_cube", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/VideoPinball-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_pinball_preprocessed() -> Self {
        Self::new(VIDEO_PINBALL_PREPROCESSED_SPEC)
    }
}
const VIDEO_PINBALL_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/VideoPinball-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_pinball", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/WizardOfWor-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn wizard_of_wor_preprocessed() -> Self {
        Self::new(WIZARD_OF_WOR_PREPROCESSED_SPEC)
    }
}
const WIZARD_OF_WOR_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/WizardOfWor-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "wizard_of_wor", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/WordZapper-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn word_zapper_preprocessed() -> Self {
        Self::new(WORD_ZAPPER_PREPROCESSED_SPEC)
    }
}
const WORD_ZAPPER_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/WordZapper-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "word_zapper", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/YarsRevenge-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn yars_revenge_preprocessed() -> Self {
        Self::new(YARS_REVENGE_PREPROCESSED_SPEC)
    }
}
const YARS_REVENGE_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/YarsRevenge-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "yars_revenge", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Zaxxon-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn zaxxon_preprocessed() -> Self {
        Self::new(ZAXXON_PREPROCESSED_SPEC)
    }
}
const ZAXXON_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Zaxxon-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "zaxxon", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

