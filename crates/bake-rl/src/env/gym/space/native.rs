//! space trait for Gymnasium environments
use bake_rl::core::env::space::{NativeSpace, DiscreteSpace, ContinuousSpace, ByteSpace};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ndarray::*;
use pyo3::{exceptions::PyValueError, prelude::*};
use burn::prelude::*;

/// `NativeGymSpace` trait which can be build from Gymnasium spaces
pub trait NativeGymSpace: NativeSpace + Sized {
    /// build the rust space from a python `gymnasium.spaces.*` object
    fn from_gym(space: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self>;
    /// traslate rust data into bounded python data
    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>>;
    /// traslate python data into rust data
    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element>;
}

impl NativeGymSpace for DiscreteSpace<crate::core::Native> {
    fn from_gym(space: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        let start: i64 = space.getattr("start")?.extract()?;
        if start != 0 {
            return Err(PyValueError::new_err(format!("Discrete(start={start}) is not supported")));
        }
        Ok(DiscreteSpace::native(space.getattr("n")?.extract()?, seed))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        Ok(data.into_pyobject(py)?.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let data: i64 = py_data.extract()?;
        Ok(data as usize)
    }
}

impl<const D: usize> NativeGymSpace for ContinuousSpace<crate::core::Native, D> {
    fn from_gym(space: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        let mut shape: Vec<usize> = vec![1];
        shape.extend_from_slice(&space.getattr("shape")?.extract::<Vec<usize>>()?);
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        
        let low = to_f32_ndarray::<D>(&space.getattr("low")?)?;
        let high = to_f32_ndarray::<D>(&space.getattr("high")?)?;
        Ok(ContinuousSpace::native(&shape, low, high, seed))
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

impl<const D: usize> NativeGymSpace for ByteSpace<crate::core::Native, D> {
    fn from_gym(space: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        let mut shape: Vec<usize> = vec![1];
        shape.extend_from_slice(&space.getattr("shape")?.extract::<Vec<usize>>()?);
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        Ok(ByteSpace::native(&shape, seed))
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
    Ok(arr.as_array().into_owned().insert_axis(Axis(0)))
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
    Ok(arr.as_array().into_owned().insert_axis(Axis(0)))
}
