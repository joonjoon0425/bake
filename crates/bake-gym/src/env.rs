//! Implementations of Gym environments
use std::marker::PhantomData;
use bake_deep::{constraint::Unconstrained, env::Environment};
use pyo3::{prelude::*, types::PyDict};
use numpy::{PyArray1, prelude::*};
use burn::prelude::*;
use crate::info::*;

/// A struct for Tabular Gymnasium environment
/// - Uses one-hot encoding
pub struct TabularGymnasiumEnvironment<I: TabularGymEnvInfo> {
    env: Py<PyAny>,
    device: Device,
    info: PhantomData<I>,
}

impl<I: TabularGymEnvInfo> TabularGymnasiumEnvironment<I> {
    /// create a new Gymnasium environment
    pub fn new(seed: u64, device: &Device, kwargs: KwArgs) -> Self {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").unwrap();
            let kwargs = kwargs.kwargs.bind(py);
            let env = gym.call_method("make", (I::name(), ), Some(&kwargs)).unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed).unwrap();
            env.call_method("reset", (), Some(&kwargs)).unwrap();
            Self {
                env: env.unbind(),
                device: device.clone(),
                info: PhantomData
            }
        });
    }

    /// return the number of observations
    pub fn n_obs(&self) -> usize { I::n_obs() }
    /// return the number of actions
    pub fn n_actions(&self) -> usize { I::n_actions() }
}

impl<I: TabularGymEnvInfo> Environment for TabularGymnasiumEnvironment<I> {
    type Obs = Tensor<2>;
    type Action = Tensor<1, Int>;
    type Constraint = Unconstrained;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let tuple = env.call_method1("reset", ()).unwrap();
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();

            let obs = obs.extract::<i64>().unwrap();
            let mut arr = vec![0f32; I::n_obs()];
            arr[obs as usize] = 1f32;
            let obs = Tensor::<1>::from_floats(arr.as_slice(), &self.device).unsqueeze_dim(0);
            (obs, Unconstrained)
        });
    }

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let action: i32 = action.into_scalar();
            let tuple = env.call_method1("step", (action, )).unwrap();
            let (obs, reward, terminate, truncate, _) : (Bound<'_, PyAny>, f32, bool, bool, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let obs = obs.extract::<i64>().unwrap();
            let mut arr = vec![0f32; I::n_obs()];
            arr[obs as usize] = 1f32;
            let obs = Tensor::<1>::from_floats(arr.as_slice(), &self.device).unsqueeze_dim(0);
            ((obs, Unconstrained), reward, terminate, truncate)
        });
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

/// A struct for Discrete Action Gymnasium environment
pub struct DiscreteGymnasiumEnvironment<const O: usize, I: DiscreteGymEnvInfo<O>> {
    env: Py<PyAny>,
    device: Device,
    info: PhantomData<I>,
}

impl<const O: usize, I: DiscreteGymEnvInfo<O>> DiscreteGymnasiumEnvironment<O, I> {
    /// create a new Gymnasium environment
    pub fn new(seed: u64, device: &Device, kwargs: KwArgs) -> Self {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").unwrap();
            let kwargs = kwargs.kwargs.bind(py);
            let env = gym.call_method("make", (I::name(), ), Some(&kwargs)).unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed).unwrap();
            env.call_method("reset", (), Some(&kwargs)).unwrap();
            Self {
                env: env.unbind(),
                device: device.clone(),
                info: PhantomData
            }
        });
    }
    /// return the shape of observation
    pub fn obs_shape(&self) -> [usize; O] { I::obs_shape() }
    /// return the number of actions
    pub fn n_actions(&self) -> usize { I::n_actions() }
}

impl<const O: usize, I: DiscreteGymEnvInfo<O>> Environment for DiscreteGymnasiumEnvironment<O, I> {
    type Obs = Tensor<O>;
    type Action = Tensor<1, Int>;
    type Constraint = Unconstrained;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let tuple = env.call_method1("reset", ()).unwrap();
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let arr = obs.cast_into::<PyArray1<f32>>().unwrap().readonly().as_array().to_vec();
            let obs: Tensor<O> = Tensor::<1>::from_floats(arr.as_slice(), &self.device).reshape(I::obs_shape());
            (obs, Unconstrained)
        })
    }

    fn step(&mut self, action: Self::Action) -> ((Self::Obs, Self::Constraint), f32, bool, bool) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let action: i32 = action.into_scalar();
            let tuple = env.call_method1("step", (action, )).unwrap();
            let (obs, reward, terminate, truncate, _) : (Bound<'_, PyAny>, f32, bool, bool, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let arr = obs.cast_into::<PyArray1<f32>>().unwrap().readonly().as_array().to_vec();
            let obs: Tensor<O> = Tensor::<1>::from_floats(arr.as_slice(), &self.device).reshape(I::obs_shape());
            ((obs, Unconstrained), reward, terminate, truncate)
        })
    }

    fn device(&self) -> Device {
        self.device.clone()
    }
}

/// A configuration for kwargs
pub struct KwArgs {
    /// keyward arguments
    pub kwargs: Py<PyDict>,
}

impl KwArgs {
    /// create a new environment config for kwargs of Gymnasium environment
    pub fn new() -> Self {
        return Python::attach(|py| {
            let kwargs = PyDict::new(py);
            Self { kwargs: kwargs.unbind() }
        });
    }

    /// add a new kwarg
    pub fn add<V>(self, kw: &'static str, arg: V) -> Self 
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

pub mod cartpole;
pub use cartpole::GymCartPole;

pub mod acrobot;
pub use acrobot::GymAcrobot;

pub mod lunarlander;
pub use lunarlander::GymLunarLander;

pub mod mountaincar;
pub use mountaincar::GymMountainCar;

pub mod taxi;
pub use taxi::GymTaxi;