//! A trait for batch dimension; the element implementing this trait can have batch dimension, and has some common operations related to batch

use std::range::Range;

/// the type of batch dimension shape
pub type BatchShape = Vec<usize>;

/// trait implementing the batch dimensions
/// # Warning
/// - The element must be endomorphic after stack, reshape, slice...
/// - The element implementing this struct must have first dimension as its 'total' batch dimension, the batch dimensions are flattened into that dimension;
/// - For example, if `burn::Tensor<2>` implements this, the `WithBatchDim::stack` operation won't really create a new axis; it actually concatenate along the first dimension
/// - But logically, since this creates a new dimension, its internal representation of batch dimension would change from empty to [n].
pub trait WithBatchDim: Sized {
    /// return current batch shape
    fn batch_shape(&self) -> BatchShape;
    /// concatanate along the given batch dimension; supports negative indexing
    /// # Panics
    /// - When other batch dimension does not matches;
    ///     - for example, given dim = 0, [1, 2, 3; feature] and [7, 2, 3; feature] would create [8, 2, 3; feature], but [1, 2, 3; feature] and [2, 3, 1, feature] would panic
    /// - When the given dimension is out of range
    fn cat(dat: Vec<Self>, dim: isize) -> Self;
    /// stack the elements, creating a new batch dimension
    /// # Panics
    /// - When the batch dimension does not matches;
    ///     - for example, [2, 3; feature] and [2, 3; feature] would create [2, 2, 3; feature], but [1, 2; feature] and [3, 2; feature] would panic
    fn stack(dat: Vec<Self>) -> Self;
    /// unstack itself along first batch dimension
    fn unstack(self) -> Vec<Self>;
    /// reshape batch dimension
    fn reshape(self, shape: BatchShape) -> Self;
    /// squeeze batch dimension; supports negative index
    /// # Panics
    /// - When the dimension of given axis is not 1
    fn squeeze(self, dim: isize) -> Self;
    /// unsqueeze batch dimension; supports negative index
    /// # Panics
    /// - When the given axis is out of range
    fn unsqueeze(self, dim: isize) -> Self;
    /// slice along given axis, with given range; supports negative index for `dim`
    fn slice(self, dim: isize, range: Range<usize>) -> Self;
    /// assign in-place the given batch, starting from the given index of given batch dimension;
    /// # Panics
    /// - When the batch dimensions other than the first (axis 0) does not matches;
    ///     - [7, 1, 3] can be assign_inplaced by [3, 1, 3], but not [7, 2, 3]
    /// - When the given data is bigger in the first batch dimension
    ///     - [7, 1, 3] cannot be assign_inplaced by [9, 1, 3] 
    fn assign_inplace(&mut self, data: Self, idx: usize);
    /// make zeros with given capacity, which has shape of given data
    fn zeros_like(capacity: usize, data: &Self) -> Self;
}

pub trait BatchIndex: Clone { fn n(&self) -> usize; }
impl BatchIndex for burn::prelude::Tensor<1, burn::prelude::Int> { fn n(&self) -> usize { self.dims()[0] } }
impl BatchIndex for Vec<usize> { fn n(&self) -> usize { self.len() } }
pub trait BatchSelect<I: BatchIndex> : WithBatchDim {
    /// gather the elements of batches of given dimension, with given indices; supports negative index for `dim`
    fn select(self, dim: isize, idx: I) -> Self;
}

mod batch_dim_impl;
pub use batch_dim_impl::{
    BatchTensor,
    BatchEmpty,
    BatchArray,
};
pub mod ops;
pub mod convert;