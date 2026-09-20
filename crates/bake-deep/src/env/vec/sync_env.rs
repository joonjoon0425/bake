//! A Synchronized Vector Environment
use crate::{data::Batchable, env::{Environment, vec::VectorizedEnvironment}};
/// A Synchronized Vector Environment implementation
pub struct SynchronizedEnvironment<E: Environment> {
    envs: Vec<E>,
}

impl<E: Environment> SynchronizedEnvironment<E> {
    /// create a new vectorized environment from given homogeneous environments
    pub fn new(envs: Vec<E>) -> Self { Self { envs } }
}

impl<E: Environment> VectorizedEnvironment<E> for SynchronizedEnvironment<E> {
    fn n_envs(&self) -> usize {
        self.envs.len()    
    }

    fn reset(&mut self, index: usize) -> (E::Obs, E::Constraint) {
        self.envs[index].reset()
    }

    fn step(&mut self, actions: <E as Environment>::Action) -> Vec<((<E as Environment>::Obs, <E as Environment>::Constraint), f32, bool, bool)> {
        let n_envs = self.n_envs();
        let mut step_result = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let action = actions.clone().slice(i..i + 1);
            step_result.push(self.envs[i].step(action))
        }
        step_result
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

            for (i, ((_, _), _, terminated, truncated)) in step_results.iter().enumerate() {
                if *terminated || *truncated {
                    println!("count: {k}, reset env no.{i}");
                    envs.reset(i);
                }
            }
        }
    }
}