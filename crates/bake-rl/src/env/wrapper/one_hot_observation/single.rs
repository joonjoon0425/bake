use crate::env::Environment;
use crate::env::space::{Space, DiscreteSpace, ContinuousSpace};
use ndarray::prelude::*;

/// one-hot encoding wrapper for discrete observation
pub struct OneHotObservation<E: Environment<ObsSpace = DiscreteSpace>> {
    env: E,
    n_obs: usize,
    obs_space: ContinuousSpace<1>
}

impl<E: Environment<ObsSpace = DiscreteSpace>> OneHotObservation<E> {
    /// create new `OneHotObservation` wrapper
    pub fn new(env: E) -> Self {
        let n_obs = env.obs_space().n();
        let low = ArrayD::from_shape_vec(vec![n_obs], vec![0.; n_obs]).unwrap();
        let high = ArrayD::from_shape_vec(vec![n_obs], vec![1.; n_obs]).unwrap();
        let obs_space = ContinuousSpace::new(&[n_obs], low, high);
        
        Self { env, obs_space, n_obs }
    }

    fn one_hot(&self, obs: usize) -> ArrayD<f32> {
        let mut vec = vec![0.; self.n_obs]; vec[obs] = 1.;
        ArrayD::from_shape_vec(vec![self.n_obs], vec).unwrap()
    }
}

impl<E: Environment<ObsSpace = DiscreteSpace>> Environment for OneHotObservation<E> {
    type ObsSpace = ContinuousSpace<1>;
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

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset();
        ((self.one_hot(o), c), i)
    }

    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        ((self.one_hot(o), c), r, ter, tru, i)
    }
}