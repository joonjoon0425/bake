//! Implementation of Gymnasium environments
use std::marker::PhantomData;

use pyo3::{prelude::*, types::PyDict};
use crate::{core::{env::{Environment, EnvironmentConfig}}, env::gym::GymSpace};
use burn::prelude::*;
/// A struct for all gymnasium environments
pub struct GymnasiumEnvironment<ObsSpace: GymSpace, ActionSpace: GymSpace> {
    env: Py<PyAny>,

    obs_space: ObsSpace,
    action_space: ActionSpace,
    device: Device,
}

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> Environment for GymnasiumEnvironment<ObsSpace, ActionSpace> {
    type ObsSpace = ObsSpace;
    type ActionSpace = ActionSpace;
    type Constraint = ();
    type Reward = Tensor<1>;
    type Info = ();

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn reset(&mut self) -> ((<Self::ObsSpace as crate::core::env::space::Space>::Element, Self::Constraint), Self::Info) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let tuple = env.call_method1("reset", ()).unwrap();
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();

            let obs = self.obs_space.from_py(&obs).unwrap();
            ((obs, ()), ())
        });
    }

    fn step(&mut self, action: <Self::ActionSpace as crate::core::env::space::Space>::Element) -> ((<Self::ObsSpace as crate::core::env::space::Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let action = self.action_space.to_py(py, action).unwrap();
            let tuple = env.call_method1("step", (action, )).unwrap();
            let (obs, reward, terminate, truncate, _) : (Bound<'_, PyAny>, f32, bool, bool, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let obs = self.obs_space.from_py(&obs).unwrap();
            let reward = Tensor::from_floats([reward], &self.device);
            ((obs, ()), reward, terminate, truncate, ())
        });
    }
}

///the configuration struct for gymnasium environments
pub struct GymnasiumEnvironmentConfig<ObsSpace: GymSpace, ActionSpace: GymSpace> {
    kwargs: Py<PyDict>,
    spec: &'static str,
    seed: usize,
    _p: PhantomData<(ObsSpace, ActionSpace)>,
}

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> GymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    /// create a new environment config for Gymnasium environment
    /// - takes `EnvSpec` of environments, in .json format
    pub fn new(spec: &'static str) -> Self {
        return Python::attach(|py| {
            let kwargs = PyDict::new(py);
            Self { kwargs: kwargs.unbind(), _p: PhantomData, spec, seed: 12 }
        });
    }

    /// set seed
    /// # Default
    /// - 12
    pub fn seed(mut self, seed: usize) -> Self {
        self.seed = seed;
        self
    }

    /// add a new kwarg
    pub fn kwarg<V>(self, kw: &'static str, arg: V) -> Self 
    where 
        V: for<'py> IntoPyObject<'py>,
    {
        return Python::attach(|py| {
            let kwargs = self.kwargs.bind(py);
            kwargs.set_item(kw, arg).unwrap();
            self
        });
    }
}

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> EnvironmentConfig for GymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    type Environment = GymnasiumEnvironment<ObsSpace, ActionSpace>;
    
    fn init(self, device: &Device) -> Self::Environment {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").expect("gymnasium module is not installed in current environment");
            let spec = py.import("gymnasium.envs.registration").expect("gymnasium.envs.registration module not found");
            let env_spec = spec.getattr("EnvSpec").expect("cannot import gymnasium.envs.registration.EnvSpec.");
            let kwargs = self.kwargs.bind(py);
            let spec = env_spec.call_method1("from_json", (self.spec, )).expect("Failed to call from_json of EnvSpec");
            let env = gym.call_method("make", (spec, ), Some(&kwargs)).unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", self.seed).unwrap();
            env.call_method("reset", (), Some(&kwargs)).unwrap();

            let obs_space = env.getattr("observation_space").unwrap();
            let obs_space = ObsSpace::from_gym(&obs_space, device).unwrap();

            let action_space = env.getattr("action_space").unwrap();
            let action_space = ActionSpace::from_gym(&action_space, device).unwrap();
            GymnasiumEnvironment {
                env: env.unbind(),
                obs_space,
                action_space,
                device: device.clone()
            }
        });
    }
}