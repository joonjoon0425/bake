//! Trait for environment

use crate::core::{data::batch_dim::convert::{AttachBatchDim, DetachBatchDim}, env::space::Space};
use burn::prelude::*;
/// `Environment` trait which all non-vectorized environments must implement
/// - For custom non-vetorized environment, implement this trait
///     - If you use `Space` struct which implements `ToBatchedSpace` trait, your environment can be easily used as vectorized environment via the `SynchronizedVectorEnvironment` wrapper.
///     - But note that it can be slower than the environment which was implemented using `VectorEnvironment` trait, since `SynchronizedVectorEnvironment` moves the data across host and device.
/// - For custom vetorized environment, implement `VectorEnvironment` trait
pub trait Environment {
    /// type of observation space
    type ObsSpace: Space;
    /// type of action space
    type ActionSpace: Space;
    /// type of constraint
    type Constraint: Clone;
    /// type of reward
    type Reward: Clone;
    /// type of additional information, which is returned every `step` and `reset`
    type Info: Clone;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environment and return the tuple ((obs, constraint), information)
    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info);
    /// return the device
    fn device(&self) -> Device;
}

/// the configuration trait for environments
pub trait EnvironmentConfig {
    /// the environment which the configuration is about
    type Environment: Environment;

    /// create a new environment, with given device
    fn init(self, device: &Device) -> Self::Environment;
}

/// trait for batchable environment
pub trait BatchableEnvironment : Environment<ObsSpace: Space<Element: AttachBatchDim>, ActionSpace: Space<Element: AttachBatchDim<Attached: DetachBatchDim>>, Constraint: AttachBatchDim, Reward: AttachBatchDim>
{}

impl<E> BatchableEnvironment for E
where
    E: Environment<ObsSpace: Space<Element: AttachBatchDim>, ActionSpace: Space<Element: AttachBatchDim<Attached: DetachBatchDim>>, Constraint: AttachBatchDim, Reward: AttachBatchDim>,
{}