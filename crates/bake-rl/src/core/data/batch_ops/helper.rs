//! helper functions for `BatchOps`
//! 

use crate::core::data::batch_ops::BatchOps;

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

/// checks if cat_batch is valid
/// # Return
/// - if cat_batch is valid, returns the catenated shape and resolved dimension
/// - if not, returns error
pub fn check_cat_batch<B: BatchOps>(bs: &[Vec<usize>], dim: isize, dat: &Vec<B>) -> Result<(Vec<usize>, usize), String> {
    if dat.is_empty() { Err("given dat vector is empty".to_string()) }
    else if dat.len() != bs.len() { Err("the number of batch shapes and dat vector does not matches".to_string()) }
    else {
        let dim = resolve_negative_index(bs[0].len(), dim)?;
        if check_batch_dim_except(&bs, dim) {
            let mut bs_res = bs[0].clone();
            let bs_dim = bs.iter().map(|bs| bs[dim]).sum::<usize>();
            bs_res[dim] = bs_dim;
            Ok((bs_res, dim))
        } else {
            Err("batch shapes does not matches".to_string())
        }
    }
}

/// check if all batch dimension equals, except given dim
pub fn check_batch_dim_except(bs: &[Vec<usize>], dim: usize) -> bool {
    let fst = &bs[0];
    bs.iter().all(
        |b|{
            if b.len() != fst.len() { return false }

            for (i, d_dim) in b.iter().enumerate() {
                if i != dim {
                    if *d_dim != fst[i] { return false }
                }
            }
            return true
        }
    )
}