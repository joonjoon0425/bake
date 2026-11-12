//! space trait for batched Gymnasium environments
use crate::core::env::space::{BatchedSpace, BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};
use super::super::convert::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use burn::prelude::*;

/// `BatchedGymSpace` trait which can be built from Gymnasium spaces
pub trait BatchedGymSpace: BatchedSpace + Sized {
    /// build the rust space from a python `gymnasium.spaces.*` object
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self>;
    /// traslate rust data into bounded python data
    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>>;
    /// traslate python data into rust data
    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element>;
}

impl BatchedGymSpace for BatchedDiscreteSpace {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let start: i64 = space.getattr("start")?.get_item(0)?.extract()?;
        if start != 0 {
            return Err(PyValueError::new_err(format!("Discrete(start={start}) is not supported")))
        }
        let n = space.getattr("nvec")?.get_item(0)?.extract()?;
        let n_envs = space.getattr("nvec")?.call_method0("__len__")?.extract()?;
        Ok(BatchedDiscreteSpace::new(n, n_envs, device))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        i32_tensor_to_numpy(py, data)
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        numpy_to_i32_tensor(py_data, &self.device())
    }
}

impl<const D: usize> BatchedGymSpace for BatchedContinuousSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let shape  = space.getattr("shape")?.extract::<Vec<usize>>()?;
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        
        let low = numpy_to_f32_tensor::<D>(&space.getattr("low")?, device)?;
        let high = numpy_to_f32_tensor::<D>(&space.getattr("high")?, device)?;
        Ok(BatchedContinuousSpace::new(&shape, low, high, device))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        f32_tensor_to_numpy(py, data)
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        numpy_to_f32_tensor(py_data, &self.device())
    }
}

impl<const D: usize> BatchedGymSpace for BatchedByteSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let shape = space.getattr("shape")?.extract::<Vec<usize>>()?;
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        Ok(BatchedByteSpace::new(&shape, device))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        u8_tensor_to_numpy(py, data)
    }


    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        numpy_to_u8_tensor(py_data, &self.device())
    }
}
