//! Synchronized vector environment wrapper
use crate::core::{data::{WithBatchDim, batch_dim::convert::{AttachBatchDim, DetachBatchDim}}, env::{BatchableEnvironment, Environment, EnvironmentConfig, VectorEnvironment, VectorEnvironmentConfig, space::Space}};
use burn::prelude::*;
/// vectorize non-vector environment in a synchronized way 
pub struct SynchronizedVectorEnvironment<E>
where 
    E: BatchableEnvironment,
{
    envs: Vec<E>,
    obs_space: E::ObsSpace,
    action_space: E::ActionSpace,
    latest_obs: Option<Vec<<<E::ObsSpace as Space>::Element as AttachBatchDim>::Attached>>,
    latest_constraint: Option<Vec<<E::Constraint as AttachBatchDim>::Attached>>,
    latest_info: Option<Vec<E::Info>>,
    device: Device
}

impl<E> VectorEnvironment for SynchronizedVectorEnvironment<E>
where
    E: BatchableEnvironment
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

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<<Self::ObsSpace as Space>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Info) {
        match mask {
            Some(mask) => {
                let mask = mask.try_into_vec_as().unwrap();
                assert!(mask.len() == self.n_envs(), "The given mask's length does not equals n_envs");

                for idx in mask.iter().enumerate().filter(|(_, b)| **b).map(|(u, _)| u) {
                    let ((obs, constraint), info) = self.envs[idx].reset();
                    self.latest_obs.as_mut().unwrap()[idx] = obs.attach_batch_dim(vec![]);
                    self.latest_constraint.as_mut().unwrap()[idx] = constraint.attach_batch_dim(vec![]);
                    self.latest_info.as_mut().unwrap()[idx] = info;
                }
                let obss = WithBatchDim::stack(self.latest_obs.clone().unwrap());
                let constraints = WithBatchDim::stack(self.latest_constraint.clone().unwrap());
                ((obss, constraints), self.latest_info.clone().unwrap())
            }
            None => {
                let mut latest_obs = Vec::with_capacity(self.n_envs());
                let mut latest_constraint = Vec::with_capacity(self.n_envs());
                let mut latest_info = Vec::with_capacity(self.n_envs());
                
                for idx in 0..self.n_envs() {
                    let ((obs, constraint), info) = self.envs[idx].reset();    
                    latest_obs.push(obs.attach_batch_dim(vec![]));
                    latest_constraint.push(constraint.attach_batch_dim(vec![]));
                    latest_info.push(info);
                }
                self.latest_obs = Some(latest_obs.clone());
                self.latest_constraint = Some(latest_constraint.clone());
                self.latest_info = Some(latest_info.clone());
                let obss = WithBatchDim::stack(latest_obs);
                let constraints = WithBatchDim::stack(latest_constraint);
                ((obss, constraints), latest_info)
            }
        }
        
    }

    fn step(&mut self, action: <<Self::ActionSpace as Space>::Element as AttachBatchDim>::Attached) -> ((<<Self::ObsSpace as Space>::Element as AttachBatchDim>::Attached, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let n_envs = self.n_envs();
        let actions = action.unstack();
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions[i].clone();
            let ((obs, constraint), reward, term, trunc, info) = self.envs[i].step(action.detach_batch_dim().0);
            self.latest_obs.as_mut().unwrap()[i] = obs.attach_batch_dim(vec![]);
            self.latest_constraint.as_mut().unwrap()[i] = constraint.attach_batch_dim(vec![]);
            self.latest_info.as_mut().unwrap()[i] = info;
            rewards.push(reward.attach_batch_dim(vec![]));
            terminated.push(term);
            truncated.push(trunc);
        }
        let rewards = WithBatchDim::stack(rewards);
        let terminated = Tensor::from_bool(terminated.as_slice(), &self.device);
        let truncated = Tensor::from_bool(truncated.as_slice(), &self.device);

        let obss = WithBatchDim::stack(self.latest_obs.clone().unwrap());
        let constraints = WithBatchDim::stack(self.latest_constraint.clone().unwrap());
        let infos= self.latest_info.clone();
        ((obss, constraints), rewards, terminated, truncated, infos.unwrap())
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

/// configuration for `SynchronizedVectorEnvironment`
pub struct SynchronizedVectorEnvironmentConfig<C>
where
    C: EnvironmentConfig<Environment: BatchableEnvironment>
{
    configs: Vec<C>,
}

impl<C> SynchronizedVectorEnvironmentConfig<C> 
where
    C: EnvironmentConfig<Environment: BatchableEnvironment>
{
    /// create a new `SynchronizedVectorEnvironmentConfig` from given configurations
    pub fn new(configs: Vec<C>) -> Self { Self { configs } }
}

impl<C> VectorEnvironmentConfig for SynchronizedVectorEnvironmentConfig<C>
where
    C: EnvironmentConfig<Environment: BatchableEnvironment>,
    SynchronizedVectorEnvironment<C::Environment> : VectorEnvironment,
{
    type Environment = SynchronizedVectorEnvironment<C::Environment>;

    fn init(self, device: &Device) -> Self::Environment {
        let envs: Vec<<C as EnvironmentConfig>::Environment> = self.configs.into_iter().map(|c| c.init(device)).collect();
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