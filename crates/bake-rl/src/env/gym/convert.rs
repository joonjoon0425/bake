use burn::prelude::*;
use ndarray::prelude::*;
use pyo3::prelude::*;
use numpy::prelude::*;
use numpy::PyReadonlyArrayDyn;

/// any array-like (list, np.ndarray of any float dtype) -> owned Tensor<D>
pub fn numpy_to_f32_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;   // already float32 -> no copy
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    let shape = arr.shape();
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), device);
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of i32 dtype) -> owned Tensor<D, Int> with Int32
pub fn numpy_to_i32_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D, Int>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "int32"))?;
    let arr: PyReadonlyArrayDyn<i32> = arr.extract()?;
    let shape = arr.shape();
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), device);
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of u8 dtype) -> owned Tensor<D, Int> with U8
pub fn numpy_to_u8_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D, Int>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "uint8"))?;
    let arr: PyReadonlyArrayDyn<u8> = arr.extract()?;
    let shape = arr.shape();
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), (device, burn::tensor::DType::U8));
    Ok(t)                               // owned, standard layout
}

/// any array-like (list, np.ndarray of u8 dtype) -> owned Tensor<D, Int> with U8
pub fn numpy_to_bool_tensor<const D: usize>(obj: &Bound<'_, PyAny>, device: &Device) -> PyResult<Tensor<D, Bool>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "bool"))?;
    let arr: PyReadonlyArrayDyn<bool> = arr.extract()?;
    let shape = arr.shape();
    let arr = arr.as_slice()?.to_vec();
    let t = Tensor::from_data(TensorData::new(arr, shape), device);
    Ok(t)                               // owned, standard layout
}

pub fn f32_tensor_to_numpy<'py, const D: usize>(py: Python<'py>, data: Tensor<D>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<f32> = data.into_data().try_into_vec::<f32>().expect("f32_tensor_to_numpy: error while converting data into Vec<f32>");
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}

pub fn i32_tensor_to_numpy<'py, const D: usize>(py: Python<'py>, data: Tensor<D, Int>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<i32> = data.into_data().try_into_vec::<i32>().expect("i32_tensor_to_numpy: error while converting data into Vec<i32>");
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}

pub fn u8_tensor_to_numpy<'py, const D: usize>(py: Python<'py>, data: Tensor<D, Int>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<u8> = data.into_data().try_into_vec::<u8>().expect("u8_tensor_to_numpy: error while converting data into Vec<u8>");
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}

pub fn bool_tensor_to_numpy<'py, const D: usize>(py: Python<'py>, data: Tensor<D, Bool>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<bool> = data.into_data().try_into_vec::<bool>().expect("bool_tensor_to_numpy: error while converting data into Vec<bool>");
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}

/// any array-like (list, np.ndarray of any float dtype) -> owned ArrayD<f32>
pub fn numpy_to_f32_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f32>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;   // already float32 -> no copy
    
    let arr: PyReadonlyArrayDyn<f32> = arr.extract()?;
    Ok(arr.as_array().into_owned())
}

/// any array-like (list, np.ndarray of any float dtype) -> owned ArrayD<u8>
pub fn numpy_to_u8_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<u8>> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "uint8"))?;   // already uint8 -> no copy
    
    let arr: PyReadonlyArrayDyn<u8> = arr.extract()?;
    Ok(arr.as_array().into_owned())
}

pub fn f32_ndarray_to_numpy<'py>(py: Python<'py>, data: ArrayD<f32>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<f32> = data.into_iter().collect();
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}

pub fn u8_ndarray_to_numpy<'py>(py: Python<'py>, data: ArrayD<u8>) -> PyResult<Bound<'py, PyAny>> {
    let shape = data.shape().to_vec();
    let vec: Vec<u8> = data.into_iter().collect();
    let py_array = vec.into_pyarray(py);
    let py_array = py_array.reshape(shape)?;
    Ok(py_array.into_any())
}