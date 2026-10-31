//! space trait for Gymnasium environments
use bake_rl::core::env::space::{Space, DiscreteSpace, ContinuousSpace, ByteSpace};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use burn::prelude::*;

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
        let data = data.detach_batch_dim().0.into_scalar::<i64>();
        Ok(data.into_pyobject(py)?.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let data: i64 = py_data.extract()?;
        Ok(Tensor::from_ints([data], &self.device()).attach_empty_dim())
    }
}

impl<const D: usize> GymSpace for ContinuousSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let mut shape: Vec<usize> = vec![1];
        shape.extend_from_slice(&space.getattr("shape")?.extract::<Vec<usize>>()?);
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        
        let low = to_f32_array(&space.getattr("low")?, device)?;
        let high = to_f32_array(&space.getattr("high")?, device)?;
        Ok(ContinuousSpace::new(&shape, low, high, device))
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        let vec: Vec<f32> = data.detach_batch_dim().0.into_data().try_into_vec_as().unwrap();
        let py_array = vec.into_pyarray(py);
        let mut shape = self.shape().to_vec();
        shape.remove(0);
        let py_array = py_array.reshape(shape)?;
        Ok(py_array.into_any())
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let arr = to_f32_array(py_data, &self.device())?;
        Ok(arr.attach_empty_dim())
    }
}

impl<const D: usize> GymSpace for ByteSpace<D> {
    fn from_gym(space: &Bound<'_, PyAny>, device: &Device) -> PyResult<Self> {
        let mut shape: Vec<usize> = vec![1];
        shape.extend_from_slice(&space.getattr("shape")?.extract::<Vec<usize>>()?);
        if shape.len() != D {
            return Err(PyValueError::new_err(format!("rank mismatch: python shape {shape:?}, rust rank {D}")))
        }
        Ok(ByteSpace::new(&shape, device))
    }

    fn from_py(&self, py_data: &Bound<'_, PyAny>) -> PyResult<Self::Element> {
        let arr = to_u8_array(py_data, &self.device())?;
        Ok(arr.attach_empty_dim())
    }

    fn to_py<'py>(&self, py: Python<'py>, data: Self::Element) -> PyResult<Bound<'py, PyAny>> {
        let vec: Vec<u8> = data.detach_batch_dim().0.into_data().try_into_vec_as().unwrap();
        let py_array = vec.into_pyarray(py);
        let mut shape = self.shape().to_vec();
        shape.remove(0);
        let py_array = py_array.reshape(shape)?;
        Ok(py_array.into_any())
    }
}

/// any array-like (list, np.ndarray of any float dtype) -> owned Tensor<D>
fn to_f32_array<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;   // already float32 -> no copy
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    let mut shape = vec![1];
    shape.extend_from_slice(arr.shape());
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), device);
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of u8 dtype) -> owned Tensor<D, Int>
fn to_u8_array<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D, Int>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "uint8"))?;
    let arr: PyReadonlyArrayDyn<u8> = arr.extract()?;
    let mut shape = vec![1];
    shape.extend_from_slice(arr.shape());
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), (device, burn::tensor::DType::U8));
    Ok(t)                               // owned, standard layout
}