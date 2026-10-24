use crate::core::{Repr, data::{WithBatchDim, batch_dim::convert::AttachBatchDim}, env::space::Space};
use burn::prelude::*;

/// `VectorEnvironment` trait which all vectorized environments must implement
/// - For custom non-vetorized environment, implement `Environment` trait
///     - You can easily make vectorized environment using the non-vectorized environment, whose space implements `ToBatchedSpace` trait, via the `SynchronizedVectorEnvironment` wrapper.
///     - But note that it can be slower than the environment which was implemented using `VectorEnvironment` trait, since `SynchronizedVectorEnvironment` moves the data across host and device.
pub trait VectorEnvironment<R: Repr> {
    /// type of observation space
    type ObsSpace: Space<R, Element: AttachBatchDim>;
    /// type of action space
    type ActionSpace: Space<R, Element: AttachBatchDim>;
    /// type of constraint
    type Constraint: WithBatchDim;
    /// type of reward
    type Reward: WithBatchDim;
    /// type of additional information, which is returned every `step` and `reset`
    type Info: Clone;

    /// return the observation space
    fn obs_space(&self) -> &Self::ObsSpace;
    /// return the action space
    fn action_space(&self) -> &Self::ActionSpace;

    /// reset the environments, according to given mask. If `None` was given, resets all environments
    /// - 0: does not reset
    /// - 1: reset
    fn reset(&mut self, mask: Option<R::Array<1, Bool>>) -> ((<<Self::ObsSpace as Space<R>>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Info);
    /// take one step and returns the tuple ((obs, constraint), reward, terminated, truncated, information)
    fn step(&mut self, action: <<Self::ActionSpace as Space<R>>::Element as AttachBatchDim>::Attached) -> ((<<Self::ObsSpace as Space<R>>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Reward, R::Array<1, Bool>, R::Array<1, Bool>, Self::Info);
    /// return the number of environments
    fn n_envs(&self) -> usize;
    /// return the device
    fn device(&self) -> Device;
}

/// configuration trait for `VectorizedEnvironment`
pub trait VectorizedEnvironmentConfig<R: Repr> {
    /// the environment which configuration is about
    type Environment: VectorEnvironment<R>;
    /// create a new environment
    fn init(self, device: &Device) -> Self::Environment;
}