//! operations related to batch dimensions

use burn::prelude::*;

/// batch-related operations
/// - most of the operations will get the batch shape as input
/// - for fixed ranks like `burn::Tensor`: regard the first dimension as flattened batch shapes
/// - for dynamic ranks: the batch shape matches the elements of tensor shape up to the length of the batch shape 
pub trait BatchOps : Sized + Clone {
    /// checks if given batch shape is compatible
    fn check_batch(&self, bs: &[usize]) -> bool;
    /// concatenate along given batch axis
    /// # Arguments
    /// - bs: batch shape of dat
    /// - dim: batch axis where data will be concatenated
    /// - dat: data to be catenated
    /// # Returns
    /// - concatenated data
    /// - batch shape after concatenation
    fn cat_batch(bs: Vec<Vec<usize>>, dim: isize, dat: Vec<Self>) -> (Self, Vec<usize>);
    /// unbind itself along given batch axis
    fn unbind_batch(self, bs: Vec<usize>, dim: isize) -> (Vec<Self>, Vec<usize>);
    /// slice along given axis, with given range
    /// # Arguments
    /// - bs: batch shape of self
    /// - dim: batch axis where self will be sliced
    /// - range: range to slice
    /// # Returns
    /// - sliced self
    /// - batch shape after slice
    fn slice_batch(self, bs: Vec<usize>, dim: isize, range: std::ops::Range<usize>) -> (Self, Vec<usize>);
    /// gather the elements of batches of given axis, with given indices;
    /// # Arguments
    /// - bs: batch shape of self
    /// - dim: batch axis where self will be selected
    /// - idx: selection index
    /// # Returns
    /// - selected self
    /// - batch shape after select
    fn select_batch(self, bs: Vec<usize>, dim: isize, idx: Tensor<1, Int>) -> (Self, Vec<usize>);
    /// assign in-place the given src, along the first batch axis, starting from idx
    fn assign_inplace(&mut self, src: Self, idx: usize);
    /// make zeros, with capacity along given batch axis
    /// # Arguments
    /// - bs: batch shape of src
    /// - dim: batch axis for capacity; that is, after operation, bs[[dim]] = capacity
    /// - capacity: the data with shape and dtype as src will have capacity along given batch axis; that is, after operation, bs[[dim]] = capacity
    /// - src: source
    /// # Returns
    /// - zero-initialized src with capacity along given batch axis
    /// - batch shape after zeros_like
    fn zeros_like_batch(bs: Vec<usize>, dim: isize, capacity: usize, src: &Self) -> (Self, Vec<usize>);
    /// enable autodiff
    fn autodiff(self) -> Self;
}

pub mod helper;
pub mod tensor;