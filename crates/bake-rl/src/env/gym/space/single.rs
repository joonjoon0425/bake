//! space trait for Gymnasium environments
use crate::env::space::{Space, DiscreteSpace, ContinuousSpace, ByteSpace};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ndarray::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use burn::prelude::*;

/// `GymSpace` trait which can be build from Gymnasium spaces
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
        
        let low = to_f32_ndarray::<D>(&space.getattr("low")?)?;
        let high = to_f32_ndarray::<D>(&space.getattr("high")?)?;
        Ok(ContinuousSpace::new(&shape, low, high))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        let vec: Vec<f32> = data.into_iter().collect();
        let py_array = vec.into_pyarray(py);
        let mut shape = self.shape().to_vec();
        shape.remove(0);
        let py_array = py_array.reshape(shape)?;
        Ok(py_array.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let arr = to_f32_ndarray::<D>(py_data)?;
        Ok(arr)
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
        let arr = to_u8_ndarray::<D>(py_data)?;
        Ok(arr)
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        let vec: Vec<u8> = data.into_iter().collect();
        let py_array = vec.into_pyarray(py);
        let mut shape = self.shape().to_vec();
        shape.remove(0);
        let py_array = py_array.reshape(shape)?;
        Ok(py_array.into_any())
    }
}

/// any array-like (list, np.ndarray of any float dtype) -> owned Tensor<D>
#[allow(unused)]
fn to_f32_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;   // already float32 -> no copy
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    let mut shape = vec![1];
    shape.extend_from_slice(arr.shape());
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), device);
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of any float dtype) -> owned ArrayD<f32>
fn to_f32_ndarray<const D: usize>(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f32>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;   // already float32 -> no copy
    
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    Ok(arr.as_array().into_owned())
}

/// any array-like (list, np.ndarray of u8 dtype) -> owned Tensor<D, Int>
#[allow(unused)]
fn to_u8_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D, Int>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "uint8"))?;
    let arr: PyReadonlyArrayDyn<u8> = arr.extract()?;
    let mut shape = vec![1];
    shape.extend_from_slice(arr.shape());
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), (device, burn::tensor::DType::U8));
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of any float dtype) -> owned ArrayD<u8>
fn to_u8_ndarray<const D: usize>(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<u8>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "uint8"))?;   // already uint8 -> no copy
    
    let arr: PyReadonlyArrayDyn<u8> = arr.extract()?;
    Ok(arr.as_array().into_owned())
}
