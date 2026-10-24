//! shape operation function and helper functions
use crate::core::data::batch_dim::{BatchIndex, BatchShape, WithBatchDim};

/// resolve the negative index; returns None if index is out of range
pub fn resolve_negative_index(n_dims: usize, dim: isize) -> Result<usize, String> {
    let n = n_dims as isize;
    let d = if dim < 0 { dim + n } else { dim };
    if !(0 <= d && d < n) {
        Err(format!("dim {dim} out of range for {n_dims} dims"))
    } else {
        Ok(d as usize)
    }
}

/// check if all batch dimension equals
pub fn check_batch_dim<T: WithBatchDim>(data: &[T]) -> bool {
    let fst_shape = data[0].batch_shape();
    data.iter().all(|d| fst_shape == d.batch_shape())
}

/// check if all batch dimension equals, except given dim
pub fn check_batch_dim_except<T: WithBatchDim>(data: &[T], dim: usize) -> bool {
    let fst_shape = data[0].batch_shape();
    data.iter().all(
        |d|{
            if d.batch_shape().len() != fst_shape.len() { return false }

            for (i, d_dim) in d.batch_shape().iter().enumerate() {
                if i != dim {
                    if *d_dim != fst_shape[i] { return false }
                }
            }
            return true
        }
    )
}

/// compute the flattened batch size
pub fn n_batch(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// checks if reshape is valid
pub fn check_reshape(from: &BatchShape, to: &BatchShape) -> bool {
    let m = from.iter().product::<usize>();
    let n = to.iter().product::<usize>();
    
    m == n
}

/// checks if squeeze is valid
/// # Return
/// - if squeeze is valid, return squeezed shape and the resolved dim
/// - else return `Err`
pub fn check_squeeze(mut shape: BatchShape, dim: isize) -> Result<(BatchShape, usize), String> {
    let dim = resolve_negative_index(shape.len(), dim)?;
    if shape[dim] != 1 {
        Err("shape[dim] != 1".into())
    }
    else {
        shape.remove(dim);
        Ok((shape, dim))
    }
}

/// checks if unsqueeze is valid
/// # Return
/// - if unsqeeuze is valid, return the unsqeezed shape and the resolved dim
/// - if not, return `Err`
pub fn check_unsqueeze(mut shape: BatchShape, dim: isize) -> Result<(BatchShape, usize), String> {
    let dim = resolve_negative_index(shape.len() + 1, dim)?;
    shape.insert(dim, 1);
    Ok((shape, dim))
}

/// checks if stack is possible
/// # Return
/// - if stack is possible, return the stacked shape
/// - if not, return `Err`
pub fn check_stack<T: WithBatchDim>(data_ref: &[T]) -> Result<BatchShape, String> {
    if data_ref.is_empty() {
        Err("Given data vector is empty".into())
    } else if !check_batch_dim(data_ref) {
        Err("The batch dimension does not matches".into())
    } else {
        let n = data_ref.len();
        let mut shape = data_ref[0].batch_shape().clone();
        shape.insert(0, n);
        Ok(shape)
    }
}

/// checks if cat is possible
/// # Return
/// - if cat is possible, return the catanated shape and resolved dim
/// - if not, return Err
pub fn check_cat<T: WithBatchDim>(dim: isize, data_ref: &[T]) -> Result<(BatchShape, usize), String> {
    if data_ref.is_empty() {
        Err("Given data vector is empty".into())
    } else {
        let mut shape = data_ref[0].batch_shape().clone();
        let dim = resolve_negative_index(shape.len(), dim)?;
        if !check_batch_dim_except(data_ref, dim) {
            return Err("The batch dimension does not matches".into())
        } 
        let mut k_final = 0;
        for d in data_ref {
            k_final += d.batch_shape()[dim];
        }
        shape[dim] = k_final;
        Ok((shape, dim))
    }
}

/// check if assign_inplace is possible
/// # Return
/// - if possible, return the total batch size, omitting the first dimension (product of all batch dimensions except axis 0);
///     - useful for flattened batch dimensions
/// - else return `Err`
pub fn check_assign_inplace(dst: &BatchShape, src: &BatchShape, idx: usize) -> Result<usize, String> {
    if dst[1..] != src[1..] {
        Err("batch dims except axis 0 must match".into())
    } else if idx + src[0] > dst[0] {
        Err("out of range".into())
    } else {
        Ok(n_batch(&dst[1..]))
    }
}

/// check if slice is possible
/// # Return
/// - if possible, return the sliced batch shape and resolved dim
/// - else return `Err`
pub fn check_slice(mut shape: BatchShape, dim: isize, range: std::range::Range<usize>) -> Result<(BatchShape, usize), String> {
    let dim = resolve_negative_index(shape.len(), dim)?;
    if range.start > range.end || range.end > shape[dim] {
        Err(format!("range {}..{} out of bounds for axis {dim} of size {}", range.start, range.end, shape[dim]))
    } else {
        shape[dim] = range.end - range.start;
        Ok((shape, dim))
    }
}

/// check if select is possible
/// # Return
/// - if possible, return the selected batch shape and resolved dim
/// - else return `Err`
pub fn check_select<I: BatchIndex>(mut shape: BatchShape, dim: isize, idx: I) -> Result<(BatchShape, usize), String> {
    let dim = resolve_negative_index(shape.len(), dim)?;
    shape[dim] = idx.n();
    Ok((shape, dim))
}