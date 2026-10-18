//! Synchronized vector environment wrapper

use crate::core::env::{Environment, EnvironmentConfig, VectorEnvironment, VectorizedEnvironmentConfig, space::{Space, to_batched::ToBatched, unbatched::batch::ToBatchedSpace}};
use burn::prelude::*;

/// vectorize non-vector environment in a synchronized way 
pub struct SynchronizedVectorEnvironment<E: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace>> {
    envs: Vec<E>,
    obs_space: <E::ObsSpace as ToBatchedSpace>::BatchedSpace,
    action_space: <E::ActionSpace as ToBatchedSpace>::BatchedSpace,
    latest_obs: Vec<<E::ObsSpace as Space>::Element>,
    latest_constraint: Vec<E::Constraint>,
    latest_info: Vec<E::Info>,
    device: Device
}

impl<E> VectorEnvironment for SynchronizedVectorEnvironment<E>
where
    E: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace>
{
    type ObsSpace = <E::ObsSpace as ToBatchedSpace>::BatchedSpace;
    type ActionSpace = <E::ActionSpace as ToBatchedSpace>::BatchedSpace;
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

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        match mask {
            Some(mask) => {
                let mask = mask.try_into_vec_as::<bool>().unwrap();
                assert!(mask.len() == self.n_envs(), "The given mask's length does not equals n_envs");

                for idx in mask.iter().enumerate().filter(|(_, b)| **b).map(|(u, _)| u) {
                    let ((obs, constraint), info) = self.envs[idx].reset();
                    self.latest_obs[idx] = obs;
                    self.latest_constraint[idx] = constraint;
                    self.latest_info[idx] = info;
                }
                let obss = E::ObsSpace::batch(self.latest_obs.clone(), &self.device);
                let constraints = E::Constraint::batch(self.latest_constraint.clone(), &self.device);
                ((obss, constraints), self.latest_info.clone())
            }
            None => {
                for idx in 0..self.n_envs() {
                    let ((obs, constraint), info) = self.envs[idx].reset();
                    self.latest_obs[idx] = obs;
                    self.latest_constraint[idx] = constraint;
                    self.latest_info[idx] = info;
                }
                let obss = E::ObsSpace::batch(self.latest_obs.clone(), &self.device);
                let constraints = E::Constraint::batch(self.latest_constraint.clone(), &self.device);
                ((obss, constraints), self.latest_info.clone())
            }
        }
        
    }

    fn step(&mut self, actions: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let n_envs = self.n_envs();
        let actions = E::ActionSpace::unbatch(actions);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions[i].clone();
            let ((obs, constraint), reward, term, trunc, info) = self.envs[i].step(action);
            self.latest_obs[i] = obs;
            self.latest_constraint[i] = constraint;
            self.latest_info[i] = info;
            rewards.push(reward);
            terminated.push(term);
            truncated.push(trunc);
        }
        let rewards = E::Reward::batch(rewards, &self.device);
        let terminated = Tensor::from_bool(terminated.as_slice(), &self.device);
        let truncated = Tensor::from_bool(truncated.as_slice(), &self.device);

        let obss = E::ObsSpace::batch(self.latest_obs.clone(), &self.device);
        let constraints = E::Constraint::batch(self.latest_constraint.clone(), &self.device);
        let infos= self.latest_info.clone();
        ((obss, constraints), rewards, terminated, truncated, infos)
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

/// configuration for `SynchronizedVectorEnvironment`
pub struct SynchronizedVectorEnvironmentConfig<C>
where
    C: EnvironmentConfig<Environment: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace>>
{
    configs: Vec<C>
}

impl<C> VectorizedEnvironmentConfig for SynchronizedVectorEnvironmentConfig<C>
where
    C: EnvironmentConfig<Environment: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace>>
{
    type Environment = SynchronizedVectorEnvironment<C::Environment>;

    fn init(self, device: &Device) -> Self::Environment {
        let mut envs: Vec<<C as EnvironmentConfig>::Environment> = self.configs.into_iter().map(|c| c.init()).collect();
        let n_envs = envs.len();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        let mut infos = Vec::with_capacity(n_envs);
        for env in &mut envs {
            let ((obs, constraint), info) = env.reset();
            obss.push(obs);
            constraints.push(constraint);
            infos.push(info);
        }
        let obs_spaces: Vec<_> = envs.iter().map(|env| env.obs_space().clone()).collect();
        let obs_space = <C::Environment as Environment>::ObsSpace::to_batched_space(obs_spaces, device);
        let action_spaces: Vec<_> = envs.iter().map(|env| env.action_space().clone()).collect();
        let action_space = <C::Environment as Environment>::ActionSpace::to_batched_space(action_spaces, device);

        Self::Environment {
            envs,
            device: device.clone(),
            latest_obs: obss,
            latest_constraint: constraints,
            latest_info: infos,
            obs_space,
            action_space
        }
    }
}