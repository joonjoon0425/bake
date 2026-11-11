use std::marker::PhantomData;

use burn::prelude::*;
use pyo3::{prelude::*, types::PyDict};

use super::super::convert::*;
use crate::core::env::{BatchedEnvironment, BatchedEnvironmentConfig};
use crate::core::env::space::BatchedSpace;
use crate::env::gym::space::BatchedGymSpace;

/// A struct for all batched gymnasium environments
pub struct BatchedGymnasiumEnvironment<ObsSpace: BatchedGymSpace, ActionSpace: BatchedSpace> {
    env: Py<PyAny>,

    n_envs: usize,

    obs_space: ObsSpace,
    action_space: ActionSpace,

    device: Device,
}

impl<ObsSpace: BatchedGymSpace, ActionSpace: BatchedGymSpace> BatchedEnvironment for BatchedGymnasiumEnvironment<ObsSpace, ActionSpace> {
    type ObsSpace = ObsSpace;
    type ActionSpace = ActionSpace;
    type Constraint = ();
    type Reward = Tensor<1>;
    type Info = ();

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn n_envs(&self) -> usize {
        self.n_envs
    }

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let tuple = match mask {
                Some(mask) => {
                    let mask = bool_tensor_to_numpy(py, mask).unwrap();
                    env.call_method1("reset", (mask, )).unwrap()
                },
                None => {
                    env.call_method1("reset", ()).unwrap()
                }
            };
            
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();

            let obs = self.obs_space.from_py(&obs).unwrap();
            ((obs, ()), ())
        });
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let action = self.action_space.to_py(py, action).unwrap();
            let tuple = env.call_method1("step", (action, )).unwrap();
            let (obs, reward, terminate, truncate, _): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let obs = self.obs_space.from_py(&obs).unwrap();
            let reward = numpy_to_f32_tensor(&reward, &self.device).unwrap();
            let terminate = numpy_to_bool_tensor(&terminate, &self.device).unwrap();
            let truncate = numpy_to_bool_tensor(&truncate, &self.device).unwrap();
            ((obs, ()), reward, terminate, truncate, ())
        });
    }
}

/// configuration struct for BatchedGymnasiumEnvironment
pub struct BatchedGymnasiumEnvironmentConfig<ObsSpace: BatchedGymSpace, ActionSpace: BatchedGymSpace> {
    kwargs: Py<PyDict>,
    spec: &'static str,
    seed: u64,
    // sync_mode: currently only Synchronized is supported
    _p: PhantomData<(ObsSpace, ActionSpace)>,
}

impl<ObsSpace: BatchedGymSpace, ActionSpace: BatchedGymSpace> BatchedGymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    /// create a new environment config for batched Gymnasium environment
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
    pub fn seed(mut self, seed: u64) -> Self {
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

impl<ObsSpace: BatchedGymSpace, ActionSpace: BatchedGymSpace> BatchedEnvironmentConfig for BatchedGymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    type Environment = BatchedGymnasiumEnvironment<ObsSpace, ActionSpace>;

    fn init(self, n_envs: usize, device: &Device) -> Self::Environment {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").expect("gymnasium module is not installed in current environment");
            let spec = py.import("gymnasium.envs.registration").expect("gymnasium.envs.registration module not found");
            let env_spec = spec.getattr("EnvSpec").expect("cannot import gymnasium.envs.registration.EnvSpec.");

            let kwargs = self.kwargs.bind(py);
            kwargs.set_item("num_envs", n_envs).unwrap();
            kwargs.set_item("vectorization_mode", "sync").unwrap();
            kwargs.set_item(
                "vector_kwargs",
                PyDict::new(py)
                .set_item(
                    "autoreset_mode",
                    gym.getattr("vector")
                    .unwrap().getattr("AutoresetMode")
                    .unwrap().getattr("DISABLED").unwrap()
                ).unwrap()).unwrap();
            let spec = env_spec.call_method1("from_json", (self.spec, )).expect("failed to call from_json of EnvSpec");

            let env = gym.call_method("make_vec", (spec, ), Some(&kwargs)).unwrap();

            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", self.seed).unwrap();
            env.call_method("reset", (), Some(&kwargs)).unwrap();

            let obs_space = env.getattr("observation_space").unwrap();
            let obs_space = ObsSpace::from_gym(&obs_space, device).unwrap();

            let action_space = env.getattr("action_space").unwrap();
            let action_space = ActionSpace::from_gym(&action_space, device).unwrap();
            
            BatchedGymnasiumEnvironment {
                env: env.unbind(),
                n_envs,
                obs_space,
                action_space,
                device: device.clone()
            }
        });
    }
}