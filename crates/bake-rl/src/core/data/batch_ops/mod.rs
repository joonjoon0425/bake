//! operations dealing with batch dimensions

/// batch dimension related operations
/// - the operations get the batch shape as input
pub trait BatchOps : Sized + 'static {
    /// checks if given batch shape is compatible
    fn check_batch(&self, bs: &[usize]) -> bool;
    /// reshape the batch dimension
    fn reshape_batch(self, bs: &[usize], new_bs: &[usize]) -> Self;
    /// concatenate along given batch axis
    /// # Arguments
    /// - bs: batch shape of dat
    /// - dim: batch axis where data will be concatenated
    /// - dat: data to be catenated
    /// # Returns
    /// - concatenated data
    fn cat_batch(bs: &[Vec<usize>], dim: i64, dat: &[&Self]) -> Self;
    /// unbind itself along given batch axis
    fn unbind_batch(self, bs: &[usize], dim: i64) -> Vec<Self>;
    /// slice along given axis, with given range
    /// # Arguments
    /// - bs: batch shape of self
    /// - dim: batch axis where self will be sliced
    /// - range: range to slice
    /// # Returns
    /// - sliced self
    fn slice_batch(self, bs: &[usize], dim: i64, range: std::ops::Range<usize>) -> Self;
    /// gather the elements of batches of given axis, with given indices;
    /// # Arguments
    /// - bs: batch shape of self
    /// - dim: batch axis where self will be selected
    /// - idx: selection index
    /// # Returns
    /// - selected self
    fn select_batch(self, bs: &[usize], dim: i64, idx: tch::Tensor) -> Self;
    /// write `src` into `self` along batch axis `dim`, starting at index `idx`
    /// # Arguments
    /// - bs: batch shape of self
    /// - dim: batch axis to write along
    /// - idx: start index along `dim`
    /// - src: data to write
    /// - src_bs: batch shape of src
    /// # Panics
    /// - if batch axes other than `dim` differ, or `idx + src_bs[dim] > bs[dim]`
    /// # Notes
    /// - the batch shape of `self` is unchanged
    fn assign_inplace_batch(&mut self, bs: &[usize], dim: i64, idx: usize, src: &Self, src_bs: &[usize]);
    /// make zeros, with capacity along given batch axis
    /// # Arguments
    /// - bs: batch shape of src
    /// - dim: batch axis for capacity; that is, after operation, bs[[dim]] = capacity
    /// - capacity: the data with shape and dtype as src will have capacity along given batch axis; that is, after operation, bs[[dim]] = capacity
    /// - src: source
    /// # Returns
    /// - zero-initialized src with capacity along given batch axis
    fn zeros_like_batch(bs: &[usize], dim: i64, capacity: usize, src: &Self) -> Self;
    /// creates an maybe-clone handle
    /// - Data type which can create a view handle creates an view handle: ex. `tch::Tensor`
    /// - Data types without a view handle calls an `clone`
    fn share(&self) -> Self;
}

mod tensor;
mod empty;
mod helper;