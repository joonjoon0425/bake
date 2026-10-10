//! Struct for converting Tabular environments into one-hot encoded observation environment
//! 
use crate::{Environment, EnvironmentConfiguration, space::{ContinuousSpace, DiscreteSpace, continuous::NdArrayContinuousSpace}};
use ndarray::prelude::*;

/// Environment for one-hot encoded observation
pub struct OneHotEncodedEnvironment<E>
where E: Environment<ObsSpace = DiscreteSpace, ActionSpace = DiscreteSpace> {
    env: E,
    obs_space: NdArrayContinuousSpace,
    action_space: DiscreteSpace
}

impl<E> OneHotEncodedEnvironment<E>
where E: Environment<ObsSpace = DiscreteSpace, ActionSpace = DiscreteSpace> {
    /// create a new one-hot encoded environment from given environment
    pub fn new(env: E) -> Self {
        let shape = vec![env.obs_space().n()];
        let obs_space = NdArrayContinuousSpace::new(
            shape.clone(),
            ArrayD::zeros(shape.clone()),
            ArrayD::ones(shape)
        );

        let action_space = DiscreteSpace::new(env.action_space().n());
        
        Self {
            env,
            obs_space,
            action_space
        }
    }
}

impl<E> Environment for OneHotEncodedEnvironment<E>
where E: Environment<ObsSpace = DiscreteSpace, ActionSpace = DiscreteSpace> {
    type ObsSpace = NdArrayContinuousSpace;
    type ActionSpace = DiscreteSpace;
    type Constraint = E::Constraint;

    fn reset(&mut self) -> (ArrayD<f32>, Self::Constraint) {
        let (obs, constraint) = self.env.reset();
        let mut onehot = ArrayD::zeros(self.obs_space.shape().clone());
        onehot[[obs]] = 1f32;
        (onehot, constraint)
    }

    fn step(&mut self, action: usize) -> ((ArrayD<f32>, Self::Constraint), f32, bool, bool) {
        let ((obs, constraint), reward, terminated, truncated) = self.env.step(action);
        let mut onehot = ArrayD::zeros(self.obs_space.shape().clone());
        onehot[obs] = 1f32;
        ((onehot, constraint), reward, terminated, truncated)
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }
}

/// Environment configuration for one-hot encoded environment
pub struct OneHotEncodedEnvironmentConfig<C: EnvironmentConfiguration> {
    config: C
}

impl<E, C: EnvironmentConfiguration<Env = E>> OneHotEncodedEnvironmentConfig<C> 
where E: Environment<ObsSpace = DiscreteSpace, ActionSpace = DiscreteSpace>
{
    /// create a new one-hot encoded environment configuration
    pub fn new(config: C) -> Self {
        Self { config }
    }
}

impl<E, C: EnvironmentConfiguration<Env = E>> EnvironmentConfiguration for OneHotEncodedEnvironmentConfig<C>
where E: Environment<ObsSpace = DiscreteSpace, ActionSpace = DiscreteSpace>
{
    type Env = OneHotEncodedEnvironment<C::Env>;
    fn init(self) -> Self::Env {
        let env = self.config.init();
        let shape = vec![env.obs_space().n()];
        let obs_space = NdArrayContinuousSpace::new(
            shape.clone(),
            ArrayD::zeros(shape.clone()),
            ArrayD::ones(shape)
        );

        let action_space = DiscreteSpace::new(env.action_space().n());
        
        OneHotEncodedEnvironment {
            env,
            obs_space,
            action_space
        }
    }
}
