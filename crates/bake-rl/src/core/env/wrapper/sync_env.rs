//! Synchronized vector environment wrapper

use std::marker::PhantomData;

use crate::core::{Repr, Vector, data::{WithBatchDim, batch_dim::convert::AttachBatchDim}, env::{BatchableEnvironment, Environment, EnvironmentConfig, VectorEnvironment, VectorizedEnvironmentConfig, space::Space}};
use burn::prelude::*;
/// vectorize non-vector environment in a synchronized way 
pub struct SynchronizedVectorEnvironment<R, E>
where 
    R: Repr,
    E: BatchableEnvironment<R>,
{
    envs: Vec<E>,
    obs_space: E::ObsSpace,
    action_space: E::ActionSpace,
    latest_obs: Option<Vec<<E::ObsSpace as Space<R>>::Element>>,
    latest_constraint: Option<Vec<E::Constraint>>,
    latest_info: Option<Vec<E::Info>>,
    device: Device
}

type Re = Vector;

impl<E> VectorEnvironment<Re> for SynchronizedVectorEnvironment<Re, E>
where
    E: BatchableEnvironment<Re>
{
    type ObsSpace = E::ObsSpace;
    type ActionSpace = E::ActionSpace;
    type Constraint = <E::Constraint as AttachBatchDim>::Attached;
    type Reward = <E::Reward as AttachBatchDim>::Attached;
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

    fn reset(&mut self, mask: Option<<Re as Repr>::Array<1, Bool>>) -> ((<<Self::ObsSpace as Space<Re>>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Info) {
        match mask {
            Some(mask) => {
                let mask = mask.try_into_vec_as().unwrap();
                assert!(mask.len() == self.n_envs(), "The given mask's length does not equals n_envs");

                for idx in mask.iter().enumerate().filter(|(_, b)| **b).map(|(u, _)| u) {
                    let ((obs, constraint), info) = self.envs[idx].reset();
                    self.latest_obs.as_mut().unwrap()[idx] = obs;
                    self.latest_constraint.as_mut().unwrap()[idx] = constraint;
                    self.latest_info.as_mut().unwrap()[idx] = info;
                }
                let obss = E::ObsSpace::batch(self.latest_obs.clone().unwrap(), &self.device);
                let constraints = E::Constraint::batch(self.latest_constraint.clone().unwrap(), &self.device);
                ((obss, constraints), self.latest_info.clone().unwrap())
            }
            None => {
                let mut latest_obs = Vec::with_capacity(self.n_envs());
                let mut latest_constraint = Vec::with_capacity(self.n_envs());
                let mut latest_info = Vec::with_capacity(self.n_envs());
                
                for idx in 0..self.n_envs() {
                    let ((obs, constraint), info) = self.envs[idx].reset();    
                    latest_obs.push(obs);
                    latest_constraint.push(constraint);
                    latest_info.push(info);
                }
                self.latest_obs = Some(latest_obs.clone());
                self.latest_constraint = Some(latest_constraint.clone());
                self.latest_info = Some(latest_info.clone());
                let obss = E::ObsSpace::batch(latest_obs, &self.device);
                let constraints = E::Constraint::batch(latest_constraint, &self.device);
                ((obss, constraints), latest_info)
            }
        }
        
    }

    fn step(&mut self, action: <<Self::ActionSpace as Space<Re>>::Element as AttachBatchDim>::Attached) -> ((<<Self::ObsSpace as Space<Re>>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let n_envs = self.n_envs();
        let actions = action.unstack();
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions[i].clone();
            let ((obs, constraint), reward, term, trunc, info) = self.envs[i].step(action);
            self.latest_obs.as_mut().unwrap()[i] = obs;
            self.latest_constraint.as_mut().unwrap()[i] = constraint;
            self.latest_info.as_mut().unwrap()[i] = info;
            rewards.push(reward);
            terminated.push(term);
            truncated.push(trunc);
        }
        let rewards = E::Reward::batch(rewards, &self.device);
        let terminated = Tensor::from_bool(terminated.as_slice(), &self.device);
        let truncated = Tensor::from_bool(truncated.as_slice(), &self.device);

        let obss = E::ObsSpace::batch(self.latest_obs.clone().unwrap(), &self.device);
        let constraints = E::Constraint::batch(self.latest_constraint.clone().unwrap(), &self.device);
        let infos= self.latest_info.clone();
        ((obss, constraints), rewards, terminated, truncated, infos.unwrap())
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

/// configuration for `SynchronizedVectorEnvironment`
pub struct SynchronizedVectorEnvironmentConfig<R, C>
where
    R: Repr,
    C: EnvironmentConfig<R, Environment: BatchableEnvironment<R>>
{
    configs: Vec<C>,
    _p: PhantomData<R>
}

impl<R, C> SynchronizedVectorEnvironmentConfig<R, C> 
where
    R: Repr,
    C: EnvironmentConfig<R, Environment: BatchableEnvironment<R>>
{
    /// create a new `SynchronizedVectorEnvironmentConfig` from given configurations
    pub fn new(configs: Vec<C>) -> Self { Self { configs, _p: PhantomData } }
}

impl<R, C> VectorizedEnvironmentConfig<R> for SynchronizedVectorEnvironmentConfig<R, C>
where
    R: Repr,
    C: EnvironmentConfig<R, Environment: BatchableEnvironment<R>>
{
    type Environment = SynchronizedVectorEnvironment<R, C::Environment>;

    fn init(self, device: &Device) -> Self::Environment {
        let envs: Vec<<C as EnvironmentConfig<R>>::Environment> = self.configs.into_iter().map(|c| c.init()).collect();
        let obs_space = envs[0].obs_space().clone();
        let action_space = envs[0].action_space().clone();

        Self::Environment {
            envs,
            device: device.clone(),
            latest_obs: None,
            latest_constraint: None,
            latest_info: None,
            obs_space,
            action_space
        }
    }
}