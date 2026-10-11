//! A Synchronized Vector Environment Wrapper
use bake_rl_core::constraint::Constraint;
use bake_rl_core::deep::constraint::convert::ToBatchable;
use bake_rl_core::deep::data::Batchable;
use crate::space::Space;
use crate::deep::space::convert::{ToBatchedSpace, ToUnbatchedSpace};
use crate::{Environment, EnvironmentConfiguration};
use crate::deep::VectorizedEnvironment;
use burn::prelude::*;
/// A Synchronized Vector Environment Wrapper implementation for Non-vectorized environments
pub struct SynchronizedEnvironment<E>
where E: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace, Constraint: Constraint + ToBatchable>
{
    envs: Vec<E>,
    device: Device,
    obs_space: <Self as VectorizedEnvironment>::ObsSpace,
    action_space: <Self as VectorizedEnvironment>::ActionSpace,
}

impl<E> SynchronizedEnvironment<E>
where E: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace, Constraint: Constraint + ToBatchable>
{
    /// create a new vectorized environment from given environment configurations
    pub fn new<C: EnvironmentConfiguration<Env = E>>(configs: Vec<C>, device: &Device) -> Self {
        let mut envs = Vec::with_capacity(configs.len());
        for config in configs {
            envs.push(config.init());
        }
        let obs_space = envs[0].obs_space().batched_space(device);
        let action_space = envs[0].action_space().batched_space(device);
        Self { envs, device: device.clone(), obs_space, action_space }
    }
}

impl<E> VectorizedEnvironment for SynchronizedEnvironment<E>
where E: Environment<ObsSpace: ToBatchedSpace, ActionSpace: ToBatchedSpace, Constraint: Constraint + ToBatchable>
{
    type ObsSpace = <E::ObsSpace as ToBatchedSpace>::BatchedSpace;
    type ActionSpace = <E::ActionSpace as ToBatchedSpace>::BatchedSpace;
    type Constraint = <E::Constraint as ToBatchable>::Batched;

    fn n_envs(&self) -> usize {
        self.envs.len()    
    }

    fn reset_all(&mut self) -> (<Self::ObsSpace as Space>::Element, Self::Constraint) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        for env in &mut self.envs {
            let (obs, constraint) = env.reset();
            let obs = self.obs_space.to_batchable(obs);
            obss.push(obs);
            constraints.push(constraint.to_batchable(&self.device));
        }
        (<Self::ObsSpace as Space>::Element::cat(obss), Self::Constraint::cat(constraints))
    }

    fn step(&mut self, actions: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Tensor<1>, Tensor<1>, Tensor<1>, (<Self::ObsSpace as Space>::Element, Self::Constraint)) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        let mut final_obss = vec![];
        let mut final_constraints = vec![];
        for i in 0..n_envs {
            let action = actions.clone().slice(i..i + 1);
            let action = self.envs[0].action_space().to_raw(action);
            let ((mut obs, mut constraint), reward, term, trunc) = self.envs[i].step(action);
            final_obss.push(self.obs_space.to_batchable(obs.clone()));
            final_constraints.push(constraint.clone().to_batchable(&self.device));
            if term || trunc { (obs, constraint) = self.envs[i].reset() }
            obss.push(self.obs_space.to_batchable(obs));
            constraints.push(constraint.to_batchable(&self.device));
            rewards.push(reward);
            terminated.push(if term { 1f32 } else { 0f32 });
            truncated.push(if trunc { 1f32 } else { 0f32 });
        }
        let rewards = Tensor::from_floats(rewards.as_slice(), &self.device());
        let terminated = Tensor::from_floats(terminated.as_slice(), &self.device());
        let truncated = Tensor::from_floats(truncated.as_slice(), &self.device());
        ((<Self::ObsSpace as Space>::Element::cat(obss), Self::Constraint::cat(constraints)), rewards, terminated, truncated, (<Self::ObsSpace as Space>::Element::cat(final_obss), Self::Constraint::cat(final_constraints)))
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use crate::{collection::CartPoleConfig, deep::{VectorizedEnvironment, wrapper::sync_env::SynchronizedEnvironment}};

    #[test]
    fn env_does_run_test() {
        let device = Device::default();
        device.seed(0);
        let seeds = [1u64, 2, 3, 4, 5];
        let mut configs = Vec::with_capacity(seeds.len());
        for seed in seeds {
            configs.push(CartPoleConfig::new().seed(seed));
        }
        let mut envs = SynchronizedEnvironment::new(configs, &device);
        envs.reset_all();
        for k in 0..50 {
            let actions = Tensor::from_ints([0, 0, 0, 0, 0], &device);
            let step_results = envs.step(actions);
            println!("iter {k}: {:?}", step_results);
        }
    }
}