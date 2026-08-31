//! A bind of Farma Foundation's Gymnasium environements
//! runs python interpreter within rust
//! 
use std::marker::PhantomData;
use pyo3::{prelude::*, types::PyDict};
use numpy::{PyArray1, prelude::*};
use burn::prelude::*;
use crate::{constraint::Unconstrained, env::Env};

pub trait GymEnvInfo {
    fn name() -> &'static str;
    fn obs_dim() -> usize;
    fn n_actions() -> usize;
}

/// ##### A Gymnasium Environment
/// Currently, only 1-ranked observation with Discrete actions are supported
pub struct GymnasiumEnv<Info: GymEnvInfo> {
    seed: u64,
    env: Py<PyAny>,
    device: Device,
    info: PhantomData<Info>,
}

impl<Info: GymEnvInfo> GymnasiumEnv<Info> {
    pub fn new(seed: u64, device: &Device, render: bool) -> Self {
        return Python::attach(|py| {
            let gym = py.import("gymnasium").unwrap();
            let kwargs = PyDict::new(py);
            if render { kwargs.set_item("render_mode", "human").unwrap(); }
            let env = gym.call_method("make", (Info::name(), ), Some(&kwargs)).unwrap();
            Self {
                seed,
                env: env.unbind(),
                device: device.clone(),
                info: PhantomData
            }
        });
        
    }

    pub fn obs_dim(&self) -> usize { Info::obs_dim() }
    pub fn n_actions(&self) -> usize { Info::n_actions() }
}

impl<Info: GymEnvInfo> Env for GymnasiumEnv<Info> {
    type Obs = Tensor<2>;
    type Action = Tensor<1, Int>;
    type Constraint = Unconstrained;

    fn reset(&mut self) -> (Self::Obs, Self::Constraint) {
        return Python::attach(|py| {
            let env = self.env.bind(py);
            let seed = PyDict::new(py);
            seed.set_item("seed", self.seed).unwrap();
            let tuple = env.call_method("reset", (), Some(&seed)).unwrap();
            let (obs, _): (Bound<'_, PyAny>, Bound<'_, PyAny>) = tuple.extract().unwrap();
            let arr = obs.cast_into::<PyArray1<f32>>().unwrap().readonly().as_array().to_vec();
            let obs: Tensor<2> = Tensor::<1>::from_floats(arr.as_slice(), &self.device).unsqueeze_dim(0);
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
            let obs: Tensor<2> = Tensor::<1>::from_floats(arr.as_slice(), &self.device).unsqueeze_dim(0);
            ((obs, Unconstrained), reward, terminate, truncate)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CartPoleInfo;
impl GymEnvInfo for CartPoleInfo {
    fn name() -> &'static str {
        "CartPole-v1"
    }

    fn obs_dim() -> usize {
        4
    }

    fn n_actions() -> usize {
        2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MountainCarInfo;
impl GymEnvInfo for MountainCarInfo {
    fn name() -> &'static str {
        "MountainCar-v0"
    }

    fn obs_dim() -> usize {
        2
    }

    fn n_actions() -> usize {
        3
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LunarLanderInfo;
impl GymEnvInfo for LunarLanderInfo {
    fn name() -> &'static str {
        "LunarLander-v3"
    }

    fn obs_dim() -> usize {
        8
    }

    fn n_actions() -> usize {
        4
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AcrobotInfo;
impl GymEnvInfo for AcrobotInfo {
    fn name() -> &'static str {
        "Acrobot-v1"
    }

    fn obs_dim() -> usize {
        6
    }

    fn n_actions() -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, GymnasiumEnv, gymnasium_env::*};

    #[test]
    fn init_and_run_environments() {
        let device = Device::default();
        device.seed(12);
        let mut env = GymnasiumEnv::<CartPoleInfo>::new(12, &device, false);
        let action = Tensor::ones([1], &Device::default());
        env.reset();
        loop {
            let ((obs, _), reward, terminated, truncated) = env.step(action.clone());
            println!("obs: {}, reward: {}, terminated: {}, truncated: {}", obs, reward, terminated, truncated);
            if terminated || truncated {
                break;
            }
        }
        
        let mut env = GymnasiumEnv::<MountainCarInfo>::new(12, &device, false);
        let action = Tensor::ones([1], &Device::default());
        env.reset();
        for _ in 0..10 {
            let ((obs, _), reward, terminated, truncated) = env.step(action.clone());
            println!("obs: {}, reward: {}, terminated: {}, truncated: {}", obs, reward, terminated, truncated);
        }

        let mut env = GymnasiumEnv::<AcrobotInfo>::new(12, &device, false);
        let action = Tensor::ones([1], &Device::default());
        env.reset();
        for _ in 0..10 {
            let ((obs, _), reward, terminated, truncated) = env.step(action.clone());
            println!("obs: {}, reward: {}, terminated: {}, truncated: {}", obs, reward, terminated, truncated);
        }

        let mut env = GymnasiumEnv::<LunarLanderInfo>::new(12, &device, false);
        let action = Tensor::ones([1], &Device::default());
        env.reset();
        for _ in 0..10 {
            let ((obs, _), reward, terminated, truncated) = env.step(action.clone());
            println!("obs: {}, reward: {}, terminated: {}, truncated: {}", obs, reward, terminated, truncated);
        }
    }
}