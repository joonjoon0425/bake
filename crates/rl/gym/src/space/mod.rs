//! A space trait for gymnasium environments
//! 

use bake_rl_env::space::{ContinuousSpace, DiscreteSpace, Space};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use numpy::ndarray::ArrayD;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Gymnasium space trait for GymnasiumEnvironments
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
        py_data.extract()
    }
}

impl<const D: usize> GymSpace for ContinuousSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape: Vec<usize> = space.getattr("shape")?.extract()?;   // tuple -> Vec
        let shape: [usize; D] = shape.as_slice().try_into().map_err(|_| {
            PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}"))
        })?;
        let low = to_f32_array(&space.getattr("low")?)?;
        let high = to_f32_array(&space.getattr("high")?)?;
        Ok(ContinuousSpace::new(shape, low, high))
    }
    fn to_py<'py>(&self, py: Python<'py>, data: ArrayD<f32>) -> PyResult<Bound<'py, PyAny>> {
        Ok(data.into_pyarray(py).into_any())                        // moves the buffer into numpy
    }
    fn from_py(&self, obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f32>> {
        let arr = to_f32_array(obj)?;
        if arr.shape() != self.shape() {
            return Err(PyValueError::new_err(format!("shape {:?} != space {:?}", arr.shape(), self.shape())));
        }
        Ok(arr)
    }
}

/// any array-like (list, np.ndarray of any float dtype) -> owned ArrayD<f32>
fn to_f32_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f32>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("asarray", (obj, "float32"))?;   // already float32 -> no copy
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    Ok(arr.as_array().to_owned())                               // owned, standard layout
}