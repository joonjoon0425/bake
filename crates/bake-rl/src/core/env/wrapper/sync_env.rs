//! Synchronized vector environment wrapper
use crate::core::{data::BatchOps, env::{Environment, VectorEnvironment, space::Space}};
use burn::prelude::*;
/// vectorize non-vector environment in a synchronized way 
pub struct SynchronizedVectorEnvironment<E>
where
    E: Environment<ObsSpace: Space<Element: BatchOps>, ActionSpace: Space<Element: BatchOps>, Constraint: BatchOps, Reward: BatchOps>
{
    envs: Vec<E>,
    obs_space: E::ObsSpace,
    action_space: E::ActionSpace,
    latest_obs: Option<Vec<<E::ObsSpace as Space>::Element>>,
    latest_constraint: Option<Vec<E::Constraint>>,
    latest_info: Option<Vec<E::Info>>,
    device: Device
}

impl<E> VectorEnvironment for SynchronizedVectorEnvironment<E>
where
    E: Environment<ObsSpace: Space<Element: BatchOps>, ActionSpace: Space<Element: BatchOps>, Constraint: BatchOps, Reward: BatchOps>
{
    type ObsSpace = E::ObsSpace;
    type ActionSpace = E::ActionSpace;
    type Constraint = E::Constraint;
    type Reward = E::Reward;
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
        let bs = vec![vec![1]; self.n_envs()];
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
                let obss = BatchOps::cat_batch(&bs, 0, self.latest_obs.clone().unwrap());
                let constraints = BatchOps::cat_batch(&bs, 0, self.latest_constraint.clone().unwrap());
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
                let obss = BatchOps::cat_batch(&bs, 0, latest_obs);
                let constraints = BatchOps::cat_batch(&bs, 0, latest_constraint);
                ((obss, constraints), latest_info)
            }
        }
        
    }

    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let n_envs = self.n_envs();
        let actions = action.unbind_batch(&[n_envs], 0);
        let bs = vec![vec![1]; n_envs];
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
        let rewards = BatchOps::cat_batch(&bs, 0, rewards);
        let terminated = Tensor::from_bool(terminated.as_slice(), &self.device);
        let truncated = Tensor::from_bool(truncated.as_slice(), &self.device);

        let obss = BatchOps::cat_batch(&bs, 0, self.latest_obs.clone().unwrap());
        let constraints = BatchOps::cat_batch(&bs, 0, self.latest_constraint.clone().unwrap());
        let infos= self.latest_info.clone();
        ((obss, constraints), rewards, terminated, truncated, infos.unwrap())
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}
