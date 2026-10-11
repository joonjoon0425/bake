//! Implementations of Gym environments
use std::marker::PhantomData;
use bake_rl_core::constraint::Unconstrained;
use bake_rl_env::{Environment, EnvironmentConfiguration};
use pyo3::{prelude::*, types::PyDict};
use crate::space::GymSpace;

/// A struct for Tabular Gymnasium environment
/// - Uses one-hot encoding
pub struct GymnasiumEnvironment<ObsSpace: GymSpace, ActionSpace: GymSpace> {
    env: Py<PyAny>,

    obs_space: ObsSpace,
    action_space: ActionSpace,
}

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> Environment for GymnasiumEnvironment<ObsSpace, ActionSpace> {
    type ObsSpace = ObsSpace;
    type ActionSpace = ActionSpace;
    type Constraint = Unconstrained<0>;

    fn reset(&mut self) -> (<Self::ObsSpace as bake_rl_env::space::Space>::Element, Self::Constraint) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let tuple = env.call_method1("reset", ()).unwrap();
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();

            let obs = self.obs_space.from_py(&obs).unwrap();
            (obs, Unconstrained)
        });
    }

    fn step(&mut self, action: <Self::ActionSpace as bake_rl_env::space::Space>::Element) -> ((<Self::ObsSpace as bake_rl_env::space::Space>::Element, Self::Constraint), f32, bool, bool) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let action = self.action_space.to_py(py, action).unwrap();
            let tuple = env.call_method1("step", (action, )).unwrap();
            let (obs, reward, terminate, truncate, _) : (Bound<'_, PyAny>, f32, bool, bool, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let obs = self.obs_space.from_py(&obs).unwrap();
            ((obs, Unconstrained), reward, terminate, truncate)
        });
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

}

/// configuration struct for gymnasium environment
pub struct GymnasiumEnvironmentConfig<ObsSpace: GymSpace, ActionSpace: GymSpace> {
    /// keyward arguments
    pub kwargs: Py<PyDict>,
    name: &'static str,
    seed: usize,
    info: PhantomData<(ObsSpace, ActionSpace)>
}

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> GymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    /// create a new environment config for Gymnasium environment
    pub fn new(name: &'static str) -> Self {
        return Python::attach(|py| {
            let kwargs = PyDict::new(py);
            Self { kwargs: kwargs.unbind(), info: PhantomData, name, seed: 12 }
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

impl<ObsSpace: GymSpace, ActionSpace: GymSpace> EnvironmentConfiguration for GymnasiumEnvironmentConfig<ObsSpace, ActionSpace> {
    type Env = GymnasiumEnvironment<ObsSpace, ActionSpace>;
    
    fn init(self) -> Self::Env {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").unwrap();
            let kwargs = self.kwargs.bind(py);
            let env = gym.call_method("make", (self.name, ), Some(&kwargs)).unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", self.seed).unwrap();
            env.call_method("reset", (), Some(&kwargs)).unwrap();

            let obs_space = env.call_method0("observation_space").unwrap();
            let obs_space = ObsSpace::from_gym(&obs_space).unwrap();

            let action_space = env.call_method0("action_space").unwrap();
            let action_space = ActionSpace::from_gym(&action_space).unwrap();
            GymnasiumEnvironment {
                env: env.unbind(),
                obs_space,
                action_space,
            }
        });
    }
}

pub mod cartpole;
pub use cartpole::cartpole;

// pub mod acrobot;
// pub use acrobot::GymAcrobot;

// pub mod lunarlander;
// pub use lunarlander::GymLunarLander;

// pub mod mountaincar;
// pub use mountaincar::GymMountainCar;

// pub mod taxi;
// pub use taxi::GymTaxi;

// pub mod cliffwalking;
// pub use cliffwalking::GymCliffWalking;

// pub mod frozenlake;
// pub use frozenlake::GymFrozenLake;