//! Synchronized batch environment wrapper
use crate::env::wrapper::into_batched::IntoBatched;
use crate::core::env::{Environment, BatchedEnvironment};
use crate::core::env::space::{Space, BatchedSpace, IntoBatchedSpace};
use burn::prelude::*;

/// batch non-batch environment in a synchronized way
/// # Note
/// - Equals that of `Gymnasium`'s `SyncVectorEnv`
pub struct SynchronizedBatchedEnvironment<E>
where
    E: Environment<ObsSpace: IntoBatchedSpace, ActionSpace: IntoBatchedSpace, Constraint: IntoBatched, Reward: IntoBatched>
{
    envs: Vec<E>,
    obs_space: <E::ObsSpace as IntoBatchedSpace>::Batched,
    action_space: <E::ActionSpace as IntoBatchedSpace>::Batched,
    latest_obs: Option<Vec<<E::ObsSpace as Space>::Element>>,
    latest_constraint: Option<Vec<E::Constraint>>,
    latest_info: Option<Vec<E::Info>>,
    device: Device
}

impl<E> SynchronizedBatchedEnvironment<E>
where
    E: Environment<ObsSpace: IntoBatchedSpace, ActionSpace: IntoBatchedSpace, Constraint: IntoBatched, Reward: IntoBatched>
{
    /// create new `SynchronizedBatchedEnvironment` from given envs 
    pub fn new(envs: Vec<E>, device: &Device) -> Self {
        assert!(!envs.is_empty(), "Given environment vector is empty");
        let (obs_spaces, action_spaces) : (Vec<E::ObsSpace>, Vec<E::ActionSpace>)= envs.iter().map(|env| (env.obs_space().clone(), env.action_space().clone())).collect();
        let obs_space = E::ObsSpace::into_batched(&obs_spaces, device);
        let action_space = E::ActionSpace::into_batched(&action_spaces, device);
        let device = device.clone();
        Self {
            envs,
            obs_space,
            action_space,
            latest_obs: None,
            latest_constraint: None,
            latest_info: None,
            device,
        }
    }
}

impl<E> BatchedEnvironment for SynchronizedBatchedEnvironment<E>
where
    E: Environment<ObsSpace: IntoBatchedSpace, ActionSpace: IntoBatchedSpace, Constraint: IntoBatched, Reward: IntoBatched>
{
    type ObsSpace = <E::ObsSpace as IntoBatchedSpace>::Batched;
    type ActionSpace = <E::ActionSpace as IntoBatchedSpace>::Batched;
    type Constraint = <E::Constraint as IntoBatched>::Batched;
    type Reward = <E::Reward as IntoBatched>::Batched;
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

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info) {
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
                let latest_obs = self.latest_obs.clone().unwrap();
                let latest_constraint = self.latest_constraint.clone().unwrap();
                let obs = <E::ObsSpace as IntoBatchedSpace>::bind(latest_obs, &self.device);
                let constraint = IntoBatched::into_batched(latest_constraint, &self.device);
                ((obs, constraint), self.latest_info.clone().unwrap())
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
                let obs = <E::ObsSpace as IntoBatchedSpace>::bind(latest_obs, &self.device);
                let constraint = IntoBatched::into_batched(latest_constraint, &self.device);
                ((obs, constraint), latest_info)
            }
        }
        
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let n_envs = self.n_envs();
        let actions = <E::ActionSpace as IntoBatchedSpace>::unbind(action);
        // assertion
        assert!(n_envs == actions.len(), "the number of environment and the number of given action does not matches");

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
        let rewards = IntoBatched::into_batched(rewards, &self.device);
        let terminated = Tensor::from_bool(terminated.as_slice(), &self.device);
        let truncated = Tensor::from_bool(truncated.as_slice(), &self.device);


        let latest_obs = self.latest_obs.clone().unwrap();
        let latest_constraint = self.latest_constraint.clone().unwrap();
        let obs = <E::ObsSpace as IntoBatchedSpace>::bind(latest_obs, &self.device);
        let constraint = IntoBatched::into_batched(latest_constraint, &self.device);
        ((obs, constraint), rewards, terminated, truncated, self.latest_info.clone().unwrap())
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}
