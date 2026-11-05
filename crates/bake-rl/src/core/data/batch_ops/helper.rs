//! helper functions for `BatchOps`
//! 

/// checks if reshape is valid
pub fn check_reshape(from: &[usize], to: &[usize]) -> bool {
    let m = from.iter().product::<usize>();
    let n = to.iter().product::<usize>();
    
    m == n
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

// helper functions made by claude

// batch-shape arithmetic for `BatchOps`
//
// Every function here is pure: it only looks at batch shapes and arguments, never at data.
// Leaves call them to validate their input, and containers call them to compute the batch
// shape after an operation.
//
// All functions resolve negative `dim` and return the resolved axis together with the shape,
// so the caller does not have to resolve it again.

use std::ops::Range;

/// resolve a possibly negative axis index
/// # Errors
/// - if `dim` is out of range for `n_dims` axes
pub fn resolve_dim(n_dims: usize, dim: i64) -> Result<usize, String> {
    let n = n_dims as i64;
    let d = if dim < 0 { dim + n } else { dim };
    if 0 <= d && d < n {
        Ok(d as usize)
    } else {
        Err(format!("dim {dim} out of range for batch shape with {n_dims} axes"))
    }
}

/// number of rows (product of the batch shape); `[]` has one row
pub fn n_rows(bs: &[usize]) -> usize {
    bs.iter().product()
}

/// batch shape after `cat` along `dim`
/// # Errors
/// - if `bs` is empty
/// - if `dim` is out of range
/// - if the shapes differ in rank or on any axis other than `dim`
/// # Returns
/// - batch shape after cat and resolved dim
pub fn bs_after_cat(bs: &[Vec<usize>], dim: i64) -> Result<(Vec<usize>, usize), String> {
    let first = bs.first().ok_or("cat: no batch shapes given")?;
    let d = resolve_dim(first.len(), dim)?;
    for (i, b) in bs.iter().enumerate() {
        if b.len() != first.len() {
            return Err(format!("cat: batch shape #{i} {b:?} has a different rank from {first:?}"));
        }
        if b.iter().zip(first).enumerate().any(|(a, (x, y))| a != d && x != y) {
            return Err(format!("cat: batch shape #{i} {b:?} differs from {first:?} outside axis {d}"));
        }
    }
    let mut out = first.clone();
    out[d] = bs.iter().map(|b| b[d]).sum();
    Ok((out, d))
}

/// batch shape after `stack` (a new leading axis of size `bs.len()`)
/// # Errors
/// - if `bs` is empty, or the shapes are not all equal
/// # Returns
/// - batch shape after stack
pub fn bs_after_stack(bs: &[Vec<usize>]) -> Result<Vec<usize>, String> {
    let first = bs.first().ok_or("stack: no batch shapes given")?;
    if let Some((i, b)) = bs.iter().enumerate().find(|(_, b)| *b != first) {
        return Err(format!("stack: batch shape #{i} {b:?} differs from {first:?}"));
    }
    let mut out = first.clone();
    out.insert(0, bs.len());
    Ok(out)
}

/// batch shape of each part after `unbind` along `dim` (the axis is removed)
/// # Errors
/// - if `bs` is empty or `dim` is out of range
/// # Returns
/// - batch shape after unbind and resolved dim
pub fn bs_after_unbind(bs: &[usize], dim: i64) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len(), dim)?;
    let mut out = bs.to_vec();
    out.remove(d);
    Ok((out, d))
}

/// batch shape after `slice` along `dim` with `range`
/// # Errors
/// - if `dim` is out of range, or `range` is reversed or exceeds the axis
/// # Returns
/// - batch shape after slice and resolved dim
pub fn bs_after_slice(bs: &[usize], dim: i64, range: &Range<usize>) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len(), dim)?;
    if range.start > range.end {
        return Err(format!("slice: reversed range {range:?}"));
    }
    if range.end > bs[d] {
        return Err(format!("slice: range {range:?} exceeds axis {d} of size {}", bs[d]));
    }
    let mut out = bs.to_vec();
    out[d] = range.end - range.start;
    Ok((out, d))
}

/// batch shape after `select` along `dim` with `n_idx` indices
/// - index values are not checked here (that would need to read the index tensor)
/// # Errors
/// - if `dim` is out of range
/// # Returns
/// - batch shape after select and resolved dim
pub fn bs_after_select(bs: &[usize], dim: i64, n_idx: usize) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len(), dim)?;
    let mut out = bs.to_vec();
    out[d] = n_idx;
    Ok((out, d))
}

/// batch shape after `zeros_like` with `capacity` along `dim`
/// # Errors
/// - if `dim` is out of range
/// # Returns
/// - batch shape after zeros_like and resolved dim
pub fn bs_after_zeros_like(bs: &[usize], dim: i64, capacity: usize) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len(), dim)?;
    let mut out = bs.to_vec();
    out[d] = capacity;
    Ok((out, d))
}

/// checks `assign` of `src_bs` into `bs` along `dim` from `idx`; the batch shape of the target is unchanged
/// # Returns
/// - the resolved axis
/// # Errors
/// - if `dim` is out of range, ranks differ, axes other than `dim` differ, or `idx + src_bs[dim] > bs[dim]`
pub fn check_assign(bs: &[usize], src_bs: &[usize], dim: i64, idx: usize) -> Result<usize, String> {
    let d = resolve_dim(bs.len(), dim)?;
    if bs.len() != src_bs.len() {
        return Err(format!("assign: batch shapes {bs:?} and {src_bs:?} have different ranks"));
    }
    if bs.iter().zip(src_bs).enumerate().any(|(a, (x, y))| a != d && x != y) {
        return Err(format!("assign: batch shapes {bs:?} and {src_bs:?} differ outside axis {d}"));
    }
    if idx + src_bs[d] > bs[d] {
        return Err(format!("assign: {} elements from index {idx} exceed axis {d} of size {}", src_bs[d], bs[d]));
    }
    Ok(d)
}

/// batch shape after `squeeze` of axis `dim`
/// # Errors
/// - if `dim` is out of range or the axis is not of size 1
/// # Returns
/// - batch shape after squeeze and resolved dim
pub fn bs_after_squeeze(bs: &[usize], dim: i64) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len(), dim)?;
    if bs[d] != 1 {
        return Err(format!("squeeze: axis {d} of {bs:?} has size {}, not 1", bs[d]));
    }
    let mut out = bs.to_vec();
    out.remove(d);
    Ok((out, d))
}

/// batch shape after `unsqueeze` at `dim` (`dim` may be `bs.len()` to append)
/// # Errors
/// - if `dim` is out of range for `bs.len() + 1` positions
/// # Returns
/// - batch shape after unsqueeze and resolved dim
pub fn bs_after_unsqueeze(bs: &[usize], dim: i64) -> Result<(Vec<usize>, usize), String> {
    let d = resolve_dim(bs.len() + 1, dim)?;
    let mut out = bs.to_vec();
    out.insert(d, 1);
    Ok((out, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve() {
        assert_eq!(resolve_dim(3, -1), Ok(2));
        assert_eq!(resolve_dim(3, 0), Ok(0));
        assert!(resolve_dim(3, 3).is_err());
        assert!(resolve_dim(3, -4).is_err());
        assert!(resolve_dim(0, 0).is_err());
    }

    #[test]
    fn cat() {
        assert_eq!(bs_after_cat(&[vec![2, 3], vec![1, 3]], 0), Ok((vec![3, 3], 0)));
        assert_eq!(bs_after_cat(&[vec![2, 3], vec![2, 5]], -1), Ok((vec![2, 8], 1)));
        assert!(bs_after_cat(&[vec![2, 3], vec![1, 4]], 0).is_err());
        assert!(bs_after_cat(&[vec![2, 3], vec![2, 3, 1]], 0).is_err());
        assert!(bs_after_cat(&[], 0).is_err());
    }

    #[test]
    fn stack() {
        assert_eq!(bs_after_stack(&[vec![3], vec![3]]), Ok(vec![2, 3]));
        assert_eq!(bs_after_stack(&[vec![], vec![], vec![]]), Ok(vec![3]));
        assert!(bs_after_stack(&[vec![3], vec![2]]).is_err());
        assert!(bs_after_stack(&[]).is_err());
    }

    #[test]
    fn unbind() {
        assert_eq!(bs_after_unbind(&[2, 3], 0), Ok((vec![3], 0)));
        assert_eq!(bs_after_unbind(&[2, 3], -1), Ok((vec![2], 1)));
        assert_eq!(bs_after_unbind(&[6], 0), Ok((vec![], 0)));
        assert!(bs_after_unbind(&[], 0).is_err());
    }

    #[test]
    fn slice() {
        assert_eq!(bs_after_slice(&[2, 3], 1, &(0..2)), Ok((vec![2, 2], 1)));
        assert_eq!(bs_after_slice(&[2, 3], -1, &(3..3)), Ok((vec![2, 0], 1)));
        assert!(bs_after_slice(&[2, 3], 1, &(1..4)).is_err());
        #[allow(clippy::reversed_empty_ranges)]
        let rev = 2..1;
        assert!(bs_after_slice(&[2, 3], 1, &rev).is_err());
    }

    #[test]
    fn select_and_zeros() {
        assert_eq!(bs_after_select(&[2, 3], 1, 5), Ok((vec![2, 5], 1)));
        assert_eq!(bs_after_zeros_like(&[2, 3], 0, 10), Ok((vec![10, 3], 0)));
        assert!(bs_after_select(&[2, 3], 2, 1).is_err());
    }

    #[test]
    fn assign() {
        assert_eq!(check_assign(&[10, 3], &[2, 3], 0, 8), Ok(0));
        assert!(check_assign(&[10, 3], &[2, 3], 0, 9).is_err());
        assert_eq!(check_assign(&[2, 5], &[2, 3], -1, 2), Ok(1));
        assert!(check_assign(&[2, 5], &[3, 3], 1, 0).is_err());
        assert!(check_assign(&[2, 5], &[2], 1, 0).is_err());
    }

    #[test]
    fn reshape_squeeze_unsqueeze() {
        assert_eq!(check_reshape(&[2, 3], &[6]), true);
        assert!(check_reshape(&[2, 3], &[4]) == false);
        assert_eq!(bs_after_squeeze(&[2, 1, 3], 1), Ok((vec![2, 3], 1)));
        assert!(bs_after_squeeze(&[2, 3], 0).is_err());
        assert_eq!(bs_after_unsqueeze(&[2, 3], 2), Ok((vec![2, 3, 1], 2)));
        assert_eq!(bs_after_unsqueeze(&[], 0), Ok((vec![1], 0)));
        assert!(bs_after_unsqueeze(&[2, 3], 3).is_err());
    }
}