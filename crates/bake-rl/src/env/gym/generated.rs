//! DO NOT EDIT
//! Auto-generated code with script/gen_gym.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/gen_gym.py`

use crate::env::gym::env::GymnasiumEnvironmentConfig;
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `CartPole-v1` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cartpole() -> Self {
        Self::new(CARTPOLE_SPEC)
    }
}
const CARTPOLE_SPEC: &'static str = r#"{"id": "CartPole-v1", "entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleEnv", "reward_threshold": 475.0, "nondeterministic": false, "max_episode_steps": 500, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv"}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, DiscreteSpace> {
    /// `MountainCar-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mountaincar() -> Self {
        Self::new(MOUNTAINCAR_SPEC)
    }
}
const MOUNTAINCAR_SPEC: &'static str = r#"{"id": "MountainCar-v0", "entry_point": "gymnasium.envs.classic_control.mountain_car:MountainCarEnv", "reward_threshold": -110.0, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `MountainCarContinuous-v0` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn mountaincar_continuous() -> Self {
        Self::new(MOUNTAINCAR_CONTINUOUS_SPEC)
    }
}
const MOUNTAINCAR_CONTINUOUS_SPEC: &'static str = r#"{"id": "MountainCarContinuous-v0", "entry_point": "gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv", "reward_threshold": 90.0, "nondeterministic": false, "max_episode_steps": 999, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

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
    pub fn lunarlander() -> Self {
        Self::new(LUNARLANDER_SPEC)
    }
}
const LUNARLANDER_SPEC: &'static str = r#"{"id": "LunarLander-v3", "entry_point": "gymnasium.envs.box2d.lunar_lander:LunarLander", "reward_threshold": 200, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ContinuousSpace<2>, ContinuousSpace<2>> {
    /// `BipedalWalker-v3` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn bipedalwalker() -> Self {
        Self::new(BIPEDALWALKER_SPEC)
    }
}
const BIPEDALWALKER_SPEC: &'static str = r#"{"id": "BipedalWalker-v3", "entry_point": "gymnasium.envs.box2d.bipedal_walker:BipedalWalker", "reward_threshold": 300, "nondeterministic": false, "max_episode_steps": 1600, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<ByteSpace<4>, ContinuousSpace<2>> {
    /// `CarRacing-v3` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn carracing() -> Self {
        Self::new(CARRACING_SPEC)
    }
}
const CARRACING_SPEC: &'static str = r#"{"id": "CarRacing-v3", "entry_point": "gymnasium.envs.box2d.car_racing:CarRacing", "reward_threshold": 900, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `Taxi-v4` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn taxi() -> Self {
        Self::new(TAXI_SPEC)
    }
}
const TAXI_SPEC: &'static str = r#"{"id": "Taxi-v4", "entry_point": "gymnasium.envs.toy_text.taxi:TaxiEnv", "reward_threshold": 8, "nondeterministic": false, "max_episode_steps": 200, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `CliffWalking-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn cliffwalking() -> Self {
        Self::new(CLIFFWALKING_SPEC)
    }
}
const CLIFFWALKING_SPEC: &'static str = r#"{"id": "CliffWalking-v1", "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

impl GymnasiumEnvironmentConfig<DiscreteSpace, DiscreteSpace> {
    /// `FrozenLake-v1` environment
    /// - `ObsSpace`: `DiscreteSpace`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frozenlak() -> Self {
        Self::new(FROZENLAK_SPEC)
    }
}
const FROZENLAK_SPEC: &'static str = r#"{"id": "FrozenLake-v1", "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv", "reward_threshold": 0.7, "nondeterministic": false, "max_episode_steps": 100, "order_enforce": true, "disable_env_checker": false, "kwargs": {"map_name": "4x4"}, "additional_wrappers": [], "vector_entry_point": null}"#;


// blackjack_flattened not implemented due to following reason: ("bake do not have `Space` correspoding to Gymnasium space of type <class 'gymnasium.spaces.tuple.Tuple'> currently",)

impl GymnasiumEnvironmentConfig<ByteSpace<4>, DiscreteSpace> {
    /// `ale_py:ALE/Pong-v5` environment
    /// - `ObsSpace`: `ByteSpace<4>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pong_preprocessed() -> Self {
        Self::new(PONG_PREPROCESSED_SPEC)
    }
}
const PONG_PREPROCESSED_SPEC: &'static str = r#"{"id": "ALE/Pong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

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
    /// `InvertedPendulum-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn inverted_pendulum() -> Self {
        Self::new(INVERTED_PENDULUM_SPEC)
    }
}
const INVERTED_PENDULUM_SPEC: &'static str = r#"{"id": "InvertedPendulum-v5", "entry_point": "gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv", "reward_threshold": 950.0, "nondeterministic": false, "max_episode_steps": 1000, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

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
    /// `Reacher-v5` environment
    /// - `ObsSpace`: `ContinuousSpace<2>`
    /// - `ActionSpace`: `ContinuousSpace<2>`
    pub fn reacher() -> Self {
        Self::new(REACHER_SPEC)
    }
}
const REACHER_SPEC: &'static str = r#"{"id": "Reacher-v5", "entry_point": "gymnasium.envs.mujoco.reacher_v5:ReacherEnv", "reward_threshold": -3.75, "nondeterministic": false, "max_episode_steps": 50, "order_enforce": true, "disable_env_checker": false, "kwargs": {}, "additional_wrappers": [], "vector_entry_point": null}"#;

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

