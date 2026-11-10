//! one-hot encoder wrapper for discrete observation
use burn::prelude::*;
use crate::core::env::BatchedEnvironment;
use crate::core::env::space::{BatchedContinuousSpace, BatchedDiscreteSpace, BatchedSpace};

/// one-hot encoding wrapper for discrete observation
pub struct BatchedOneHotObservation<E: BatchedEnvironment<ObsSpace = BatchedDiscreteSpace>> {
    env: E,
    n_obs: usize,
    obs_space: BatchedContinuousSpace<2>
}

impl<E: BatchedEnvironment<ObsSpace = BatchedDiscreteSpace>> BatchedOneHotObservation<E> {
    /// create new `OneHotObservation` wrapper
    pub fn new(env: E) -> Self {
        let n_obs = env.obs_space().n();
        let device = env.device();
        let low = Tensor::from_data(TensorData::new(vec![0; n_obs], [1, n_obs]), &device);
        let high = Tensor::from_data(TensorData::new(vec![1; n_obs], [1, n_obs]), &device);
        let obs_space = BatchedContinuousSpace::new(&[1, n_obs], low, high, &device);
        
        Self { env, obs_space, n_obs }
    }

    fn one_hot(&self, obs: Tensor<1, Int>) -> Tensor<2> {
        obs.one_hot(self.n_obs).float()
    }
}

impl<E: BatchedEnvironment<ObsSpace = BatchedDiscreteSpace>> BatchedEnvironment for BatchedOneHotObservation<E> {
    type ObsSpace = BatchedContinuousSpace<2>;
    type ActionSpace = E::ActionSpace;
    type Constraint = E::Constraint;
    type Reward = E::Reward;
    type Info = E::Info;

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        self.env.action_space()
    }

    fn device(&self) -> Device {
        self.env.device()
    }

    fn n_envs(&self) -> usize {
        self.env.n_envs()
    }

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset(mask);
        ((self.one_hot(o), c), i)
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        ((self.one_hot(o), c), r, ter, tru, i)
    }
}