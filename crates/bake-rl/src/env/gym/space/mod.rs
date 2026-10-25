//! space trait for Gymnasium environments
use bake_rl::core::env::space::{Space, DiscreteSpace};
use pyo3::{exceptions::PyValueError, prelude::*};
use burn::prelude::*;

use crate::core::env::space::ContinuousSpace;

/// `GymSpace` trait which can be build from Gymnasium spaces
pub trait GymSpace: Space + Sized {
    /// build the rust space from a python `gymnasium.spaces.*` object
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self>;
    /// traslate rust data into bounded python data
    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>>;
    /// traslate python data into rust data
    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element>;
}

impl GymSpace for DiscreteSpace {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let start: i64 = space.getattr("start")?.extract()?;
        if start != 0 {
            return Err(PyValueError::new_err(format!("Discrete(start={start}) is not supported")));
        }
        Ok(DiscreteSpace::new(space.getattr("n")?.extract()?, device))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        let data = data.into_scalar::<i64>();
        Ok(data.into_pyobject(py)?.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let data: i64 = py_data.extract()?;
        Ok(Tensor::from_ints([data], &self.device()))
    }
}

impl<const D: usize> GymSpace for ContinuousSpace<D> {
    
}