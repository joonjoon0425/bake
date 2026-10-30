//! helper functions for `BatchOps`
//! 

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

/// checks if reshape is valid
pub fn check_reshape(from: &Vec<usize>, to: &Vec<usize>) -> bool {
    let m = from.iter().product::<usize>();
    let n = to.iter().product::<usize>();
    
    m == n
}

/// # Return
/// - if squeeze is valid, return squeezed shape and the resolved dim
/// - else return `Err`
pub fn check_squeeze(mut shape: Vec<usize>, dim: isize) -> Result<(Vec<usize>, usize), String> {
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
pub fn check_unsqueeze(mut shape: Vec<usize>, dim: isize) -> Result<(Vec<usize>, usize), String> {
    let dim = resolve_negative_index(shape.len() + 1, dim)?;
    shape.insert(dim, 1);
    Ok((shape, dim))
}
