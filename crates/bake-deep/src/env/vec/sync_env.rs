//! A Synchronized Vector Environment Wrapper
use crate::{data::Batchable, env::{Environment, vec::VectorizedEnvironment}};
use burn::prelude::*;
/// A Synchronized Vector Environment Wrapper implementation for Non-vectorized environments
pub struct SynchronizedEnvironment<E: Environment> {
    envs: Vec<E>,
    final_obss: Vec<E::Obs>,
    final_constraints: Vec<E::Constraint>,
}

impl<E: Environment> SynchronizedEnvironment<E> {
    /// create a new vectorized environment from given homogeneous environments
    pub fn new(envs: Vec<E>) -> Self {
        Self { envs, final_obss: vec![], final_constraints: vec![] }
    }
}

impl<E: Environment> VectorizedEnvironment for SynchronizedEnvironment<E> {
    type Obs = E::Obs;
    type Action = E::Action;
    type Constraint = E::Constraint;

    fn n_envs(&self) -> usize {
        self.envs.len()    
    }

    fn reset_all(&mut self) -> (Self::Obs, Self::Constraint) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        for env in &mut self.envs {
            let (obs, constraint) = env.reset();
            obss.push(obs);
            constraints.push(constraint);
        }
        (Self::Obs::cat(obss), Self::Constraint::cat(constraints))
    }

    fn step(&mut self, actions: <E as Environment>::Action) -> ((<E as Environment>::Obs, <E as Environment>::Constraint), Tensor<1>, Tensor<1>, Tensor<1>, (<E as Environment>::Obs, <E as Environment>::Constraint)) {
        let n_envs = self.n_envs();
        let mut obss = Vec::with_capacity(n_envs);
        let mut constraints = Vec::with_capacity(n_envs);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut terminated = Vec::with_capacity(n_envs);
        let mut truncated = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions.clone().slice(i..i + 1);
            let ((mut obs, mut constraint), reward, term, trunc) = self.envs[i].step(action);
            self.final_obss.push(obs.clone());
            self.final_constraints.push(constraint.clone());
            if term || trunc { (obs, constraint) = self.envs[i].reset() }
            obss.push(obs);
            constraints.push(constraint);
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
        envs.reset_all();
        for k in 0..50 {
            let actions = Tensor::from_ints([0, 0, 0, 0, 0], &device);
            let step_results = envs.step(actions);
            println!("iter {k}: {:?}", step_results);
        }
    }
}