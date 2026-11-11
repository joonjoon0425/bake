//! space trait for Gymnasium environments
use crate::core::env::space::{Space, DiscreteSpace, ContinuousSpace, ByteSpace};
use super::super::convert::*;
use pyo3::{exceptions::PyValueError, prelude::*};

/// `GymSpace` trait which can be built from Gymnasium spaces
pub trait GymSpace: Space + Sized {
    /// build the rust space from a python `gymnasium.spaces.*` object
    fn from_gym(space: &Bound<'_, PyAny>) -> PyResult<Self>;
    /// traslate rust data into bounded python data
    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>>;
    /// traslate python data into rust data
    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element>;
}

impl GymSpace for DiscreteSpace {
    fn from_gym(space: &Bound<'_, PyAny>) -> PyResult<Self> {
        let start: i64 = space.getattr("start")?.extract()?;
        if start != 0 {
            return Err(PyValueError::new_err(format!("Discrete(start={start}) is not supported")));
        }
        Ok(DiscreteSpace::new(space.getattr("n")?.extract()?))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        Ok(data.into_pyobject(py)?.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let data: i64 = py_data.extract()?;
        Ok(data as usize)
    }
}

impl<const D: usize> GymSpace for ContinuousSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape  = space.getattr("shape")?.extract::<Vec<usize>>()?;
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        
        let low = numpy_to_f32_ndarray(&space.getattr("low")?)?;
        let high = numpy_to_f32_ndarray(&space.getattr("high")?)?;
        Ok(ContinuousSpace::new(&shape, low, high))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        f32_ndarray_to_numpy(py, data)
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        numpy_to_f32_ndarray(py_data)
    }
}

impl<const D: usize> GymSpace for ByteSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = space.getattr("shape")?.extract::<Vec<usize>>()?;
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        Ok(ByteSpace::new(&shape))
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        numpy_to_u8_ndarray(py_data)
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        u8_ndarray_to_numpy(py, data)
    }
}