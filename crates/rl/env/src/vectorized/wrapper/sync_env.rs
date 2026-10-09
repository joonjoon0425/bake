//! A Synchronized Vector Environment Wrapper
use bake_rl_core::deep::data::Batchable;
use crate::vectorized::element::BatchableElement;
use crate::{Environment, EnvironmentConfiguration};
use crate::vectorized::VectorizedEnvironment;
use burn::prelude::*;
/// A Synchronized Vector Environment Wrapper implementation for Non-vectorized environments
pub struct SynchronizedEnvironment<E>
where E: Environment<Obs: BatchableElement, Action: BatchableElement, Constraint: BatchableElement>
{
    envs: Vec<E>,
    device: Device,
    final_obss: Vec<<E::Obs as BatchableElement>::Batched>,
    final_constraints: Vec<<E::Constraint as BatchableElement>::Batched>,
}

impl<E> SynchronizedEnvironment<E>
where E: Environment<Obs: BatchableElement, Action: BatchableElement, Constraint: BatchableElement>
{
    /// create a new vectorized environment from given environment configurations
    pub fn new<C: EnvironmentConfiguration<Env = E>>(configs: Vec<C>, device: &Device) -> Self {
        let mut envs = Vec::with_capacity(configs.len());
        for config in configs {
            envs.push(config.init());
        }
        Self { envs, device: device.clone(), final_obss: vec![], final_constraints: vec![]}
    }
}

impl<E> VectorizedEnvironment for SynchronizedEnvironment<E> 
where E: Environment<Obs: BatchableElement, Action: BatchableElement, Constraint: BatchableElement>
{
    type Obs = <E::Obs as BatchableElement>::Batched;
    type Action = <E::Action as BatchableElement>::Batched;
    type Constraint = <E::Constraint as BatchableElement>::Batched;

    fn n_envs(&self) -> usize {
        self.envs.len()    
    }

    fn reset_all(&mut self) -> (Self::Obs, Self::Constraint) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        for env in &mut self.envs {
            let (obs, constraint) = env.reset();
            obss.push(obs.to_batchable(&self.device));
            constraints.push(constraint.to_batchable(&self.device));
        }
        (Self::Obs::cat(obss), Self::Constraint::cat(constraints))
    }

    fn step(&mut self, actions: Self::Action) -> ((Self::Obs, Self::Constraint), Tensor<1>, Tensor<1>, Tensor<1>, (Self::Obs, Self::Constraint)) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions.clone().slice(i..i + 1);
            let ((mut obs, mut constraint), reward, term, trunc) = self.envs[i].step(E::Action::to_raw(action));
            self.final_obss.push(obs.clone().to_batchable(&self.device));
            self.final_constraints.push(constraint.clone().to_batchable(&self.device));
            if term || trunc { (obs, constraint) = self.envs[i].reset() }
            obss.push(obs.to_batchable(&self.device));
            constraints.push(constraint.to_batchable(&self.device));
            rewards.push(reward);
            terminated.push(if term { 1f32 } else { 0f32 });
            truncated.push(if trunc { 1f32 } else { 0f32 });
        }
        let rewards = Tensor::from_floats(rewards.as_slice(), &self.device());
        let terminated = Tensor::from_floats(terminated.as_slice(), &self.device());
        let truncated = Tensor::from_floats(truncated.as_slice(), &self.device());
        let final_obss = self.final_obss.clone();
        let final_constraints = self.final_constraints.clone();
        self.final_obss.clear();
        self.final_constraints.clear();
        ((Self::Obs::cat(obss), Self::Constraint::cat(constraints)), rewards, terminated, truncated, (Self::Obs::cat(final_obss), Self::Constraint::cat(final_constraints)))
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use crate::{collection::CartPoleConfig, vectorized::{VectorizedEnvironment, wrapper::sync_env::SynchronizedEnvironment}};

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