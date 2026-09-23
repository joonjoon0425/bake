//! A Synchronized Vector Environment Wrapper
use crate::{data::Batchable, env::{Environment, vec::VectorizedEnvironment}};
use burn::prelude::*;
/// A Synchronized Vector Environment Wrapper implementation for Non-vectorized environments
pub struct SynchronizedEnvironment<E: Environment> {
    envs: Vec<E>,
}

impl<E: Environment> SynchronizedEnvironment<E> {
    /// create a new vectorized environment from given homogeneous environments
    pub fn new(envs: Vec<E>) -> Self { Self { envs } }
}

impl<E: Environment> VectorizedEnvironment for SynchronizedEnvironment<E> {
    type Obs = E::Obs;
    type Action = E::Action;
    type Constraint = E::Constraint;

    fn n_envs(&self) -> usize {
        self.envs.len()    
    }

    fn reset(&mut self, indices: Tensor<1>) -> (E::Obs, E::Constraint) {
        
    }

    fn step(&mut self, actions: <E as Environment>::Action) -> ((<E as Environment>::Obs, <E as Environment>::Constraint), Tensor<1>, Tensor<1>, Tensor<1>) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions.clone().slice(i..i + 1);
            let ((obs, constraint), reward, term, trunc) = self.envs[i].step(action);
            obss.push(obs);
            constraints.push(constraint);
            rewards.push(reward);
            terminated.push(if term { 1f32 } else { 0f32 });
            truncated.push(if trunc { 1f32 } else { 0f32 });
        }
        let rewards = Tensor::from_floats(rewards.as_slice(), &self.device());
        let terminated = Tensor::from_floats(terminated.as_slice(), &self.device());
        let truncated = Tensor::from_floats(truncated.as_slice(), &self.device());
        ((Self::Obs::cat(obss), Self::Constraint::cat(constraints)), rewards, terminated, truncated)
    }

    fn device(&self) -> burn::prelude::Device {
        self.envs[0].device()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use crate::env::{CartPole, vec::{SynchronizedEnvironment, VectorizedEnvironment}};

    #[test]
    fn env_does_run_test() {
        let device = Device::default();
        device.seed(0);
        let seeds = [1u64, 2, 3, 4, 5];
        let mut envs = Vec::with_capacity(seeds.len());
        for seed in seeds {
            envs.push(CartPole::new(seed, &device));
        }
        let mut envs = SynchronizedEnvironment::new(envs);
        for i in 0..seeds.len() { envs.reset(i); }
        for k in 0..50 {
            let actions = Tensor::from_ints([0, 0, 0, 0, 0], &device);
            let step_results = envs.step(actions);
            println!("iter {k}: {:?}", step_results);
        }
    }
}