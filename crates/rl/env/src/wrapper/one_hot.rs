//! Struct for converting Tabular environments into one-hot encoded observation environment
//! 
use crate::{DiscreteEnvironment, Environment, EnvironmentConfiguration, TabularEnvironment};

/// Environment for one-hot encoded observation
pub struct OneHotEncodedEnvironment<E>
where E: TabularEnvironment {
    env: E,
}

impl<E> OneHotEncodedEnvironment<E>
where E: TabularEnvironment {
    /// create a new one-hot encoded environment from given environment
    pub fn new(env: E) -> Self { Self { env } }
}

impl<E> Environment for OneHotEncodedEnvironment<E>
where E: TabularEnvironment {
    type Obs = Vec<f32>;
    type Action = E::Action;
    type Constraint = E::Constraint;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        let (obs, constraint) = self.env.reset();
        let mut onehot = vec![0f32; self.env.n_obs()]; onehot[obs] = 1f32;
        (onehot, constraint)
    }

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        let ((obs, constraint), reward, terminated, truncated) = self.env.step(action);
        let mut onehot = vec![0f32; self.env.n_obs()]; onehot[obs] = 1f32;
        ((onehot, constraint), reward, terminated, truncated)
    }
}

impl<E> DiscreteEnvironment for OneHotEncodedEnvironment<E>
where E: TabularEnvironment {
    fn obs_shape(&self) -> Vec<usize> {
        vec![self.env.n_obs()]
    }

    fn n_actions(&self) -> usize {
        self.env.n_actions()    
    }

    fn obs_range(&self) -> (Self::Obs, Self::Obs) {
        (vec![0f32; self.env.n_obs()], vec![1f32; self.env.n_obs()])
    }
}
/// Environment configuration for one-hot encoded environment
pub struct OneHotEncodedEnvironmentConfig<C: EnvironmentConfiguration> {
    config: C
}

impl<C: EnvironmentConfiguration<Env: TabularEnvironment>> OneHotEncodedEnvironmentConfig<C> {
    /// create a new one-hot encoded environment configuration
    pub fn new(config: C) -> Self {
        Self { config }
    }
}

impl<C: EnvironmentConfiguration<Env: TabularEnvironment>> EnvironmentConfiguration for OneHotEncodedEnvironmentConfig<C> {
    type Env = OneHotEncodedEnvironment<C::Env>;
    fn init(self) -> Self::Env {
        OneHotEncodedEnvironment { env: self.config.init() }
    }
}
