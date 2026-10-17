//! Synchronized vector environment wrapper

use crate::core::env::{Environment, EnvironmentConfig, VectorEnvironment, space::{Space, switch::ToBatched, unbatched::batch::ToBatchedSpace}};
use crate::core::data::Batchable;
use burn::prelude::*;

/// vectorize non-vector environment in a synchronized way 
pub struct SynchronizedVectorEnvironment<E: Environment + ToBatchedSpace> {
    envs: Vec<E>,
    obs_space: <E as ToBatchedSpace>::BatchedSpace,
    action_space: <E as ToBatchedSpace>::BatchedSpace,
    latest_obs: Vec<<E::ObsSpace as Space>::Element>,
    latest_constraint: Vec<E::Constraint>,
    latest_info: Vec<E::Info>,
    device: Device
}

impl<E> VectorEnvironment for SynchronizedVectorEnvironment<E>
where
    E: Environment + ToBatchedSpace
{
    type ObsSpace = <E as ToBatchedSpace>::BatchedSpace;
    type ActionSpace = <E as ToBatchedSpace>::BatchedSpace;
    type Constraint = <E::Constraint as ToBatched>::Batched;
    type Reward = <E::Reward as ToBatched>::Batched;
    type Info = Vec<E::Info>;

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn n_envs(&self) -> usize {
        self.envs.len()
    }

    fn reset(&mut self, mask: Tensor<1, Bool>) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        let n_envs = self.n_envs();
        let mask = mask.try_into_vec_as::<bool>().unwrap();
        assert!(mask.len() == self.n_envs(), "The given mask's length does not equals n_envs");

        for idx in mask.iter().enumerate().filter(|(_, &b)| b).map(|(u, _)| u) {
            let ((obs, constraint), info) = self.envs[idx].reset();
            self.latest_obs[idx] = obs;
            self.latest_constraint[idx] = constraint;
            self.latest_info[idx] = info;
        }
        E::ObsSpace::batch(self.latest_obs, &self.device)
        ((<Self::ObsSpace as Space>::Element::cat(self.latest_obs), Self::Constraint::cat(self.latest_constraint)), self.latest_info)
    }
}

/// configuration for `SynchronizedVectorEnvironment`
pub struct SynchronizedVectorEnvironmentConfig<C: EnvironmentConfig> {
    configs: Vec<C>
}