//! A trait for batch dimension; the element implementing this trait can have batch dimension, and has some common operations related to batch

use std::range::Range;

type Shape = Vec<usize>;

/// trait implementing the batch dimensions
/// # Warning
/// - The element must be endomorphic after stack, reshape, slice...
/// - The element implementing this struct must have first dimension as its batch dimension, and it must be 1-dimensional;
/// - For example, if `burn::Tensor<2>` implements this, the `WithBatchDim::stack` operation won't really create a new axis; it actually concatenate along the first dimension
/// - But logically, since this creates a new dimension, its internal representation of batch dimension would change from empty to [n].
pub trait WithBatchDim: Sized {
    /// return current batch shape
    fn batch_shape(&self) -> &Shape;
    /// concatanate along the given batch dimension; supports negative indexing
    /// # Panics
    /// - When other batch dimension does not matches;
    ///     - for example, given dim = 0, [1, 2, 3; feature] and [7, 2, 3; feature] would create [8, 2, 3; feature], but [1, 2, 3; feature] and [2, 3, 1, feature] would panic
    /// - When the given dimension is out of range
    fn cat(dim: isize, dat: Vec<Self>) -> Self;
    /// stack the elements, creating a new batch dimension
    /// # Panics
    /// - When the batch dimension does not matches;
    ///     - for example, [2, 3; feature] and [2, 3; feature] would create [2, 2, 3; feature], but [1, 2; feature] and [3, 2; feature] would panic
    fn stack(dat: Vec<Self>) -> Self;
    /// reshape batch dimension
    fn reshape(self, shape: Shape) -> Self;
    /// squeeze batch dimension; supports negative index
    /// # Panic
    /// - When the dimension of given axis is not 1
    fn squeeze(self, dim: isize) -> Self;
    /// unsqueeze batch dimension; supports negative index
    /// # Panic
    /// - When the given axis is out of range
    fn unsqueeze(self, dim: isize) -> Self;
    /// slice along given axis, with given range
    fn slice(self, dim: isize, range: Range<usize>) -> Self;
}