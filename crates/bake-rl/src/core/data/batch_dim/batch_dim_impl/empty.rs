use std::marker::PhantomData;

use crate::core::data::batch_dim::{BatchIndex, BatchSelect, BatchShape, WithBatchDim};
use crate::core::data::batch_dim::ops::*;

/// `Unconstrained` or any empty struct with batch dimension
/// - I may change return type of `batch_size()` into `Option` and delete this struct
#[derive(Debug, Clone)]
pub struct BatchEmpty<T: Clone> {
    shape: BatchShape,
    _p: PhantomData<T>
}

impl<T: Clone> BatchEmpty<T> {
    /// create a new `BatchEmpty<T>` struct with empty batch dimension
    pub fn with_empty_batch_dim() -> Self {
        Self {
            shape: vec![],
            _p: PhantomData
        }
    }

    /// create a new `BatchEmpty<T>` struct with given batch dimension
    pub fn with_batch_dim(batch_shape: BatchShape) -> Self {
        Self {
            shape: batch_shape,
            _p: PhantomData
        }
    }
}

impl<T: Clone> WithBatchDim for BatchEmpty<T> {
    fn batch_shape(&self) -> BatchShape {
        self.shape.clone()
    }
    fn reshape(self, shape: BatchShape) -> Self {
        assert!(check_reshape(&self.shape, &shape), "invalid shape given");
        Self { shape, _p: PhantomData }
    }

    fn squeeze(self, dim: isize) -> Self {
        let (shape, _) = check_squeeze(self.shape, dim).unwrap();
        Self {
            shape,
            ..self
        }
    }

    fn unsqueeze(self, dim: isize) -> Self {
        let (shape, _) = check_unsqueeze(self.shape, dim).unwrap();
        Self {
            shape,
            ..self
        }
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        check_assign_inplace(&self.shape, &data.shape, idx).unwrap();
    }

    fn stack(dat: Vec<Self>) -> Self {
        let shape = check_stack(&dat).unwrap();
        Self {
            shape,
            _p: PhantomData
        }
    }

    fn unstack(mut self) -> Vec<Self> {
        assert!(!self.shape.is_empty(), "unstack: BatchEmpty<T> has empty batch dimension");
        let n = self.shape.remove(0);
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(BatchEmpty { shape: self.shape.clone(), _p: PhantomData })
        }

        vec
    }
    
    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        let (shape, _) = check_cat(dim, &dat).unwrap();
        Self {
            shape,
            _p: PhantomData
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let (shape, _) = check_slice(self.shape, dim, range).unwrap();
        Self {
            shape,
            _p: PhantomData
        }
    }
    
    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let mut shape = data.shape.clone();
        shape.insert(0, capacity);

        Self {
            shape,
            _p: PhantomData
        }
    }
}

impl<T: Clone, I: BatchIndex> BatchSelect<I> for BatchEmpty<T> {
    fn select(self, dim: isize, idx: I) -> Self {
        let (shape, _) = check_select(self.shape, dim, idx).unwrap();
        Self {
            shape,
            _p: PhantomData
        }
    }
}

// tests made by Claude

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use crate::core::data::batch_dim::{BatchSelect, BatchEmpty, BatchShape, WithBatchDim};

    fn d() -> Device { Device::default() }
    fn be(shape: BatchShape) -> BatchEmpty<()> { BatchEmpty::<()>::with_empty_batch_dim().with_shape_for_test(shape) }

    impl<T: Clone> BatchEmpty<T> {
        fn with_shape_for_test(mut self, shape: BatchShape) -> Self { self.shape = shape; self }
    }

    // ───────────── new / 메타데이터 ─────────────

    #[test]
    fn new_is_empty_batch() {
        let b = BatchEmpty::<()>::with_empty_batch_dim();
        assert!(b.batch_shape().is_empty());
    }

    #[test]
    fn reshape() {
        let b = be(vec![2, 3]).reshape(vec![3, 2]);
        assert_eq!(vec![3, 2], *b.batch_shape());
    }

    #[test]
    fn reshape_from_empty_batch() {
        // [] 의 원소 수는 1 (빈 곱)
        let b = be(vec![]).reshape(vec![1, 1]);
        assert_eq!(vec![1, 1], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn reshape_panic() {
        be(vec![2, 3]).reshape(vec![4]);
    }

    #[test]
    fn squeeze() {
        let b = be(vec![2, 3, 4, 1]).squeeze(3);
        assert_eq!(vec![2, 3, 4], *b.batch_shape());
    }

    #[test]
    fn squeeze_neg_idx() {
        let b = be(vec![2, 1, 2]).squeeze(-2);
        assert_eq!(vec![2, 2], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn squeeze_panic_not_1() {
        be(vec![2, 1]).squeeze(0);
    }

    #[test]
    #[should_panic]
    fn squeeze_empty_batch_panic() {
        be(vec![]).squeeze(-1);
    }

    #[test]
    fn unsqueeze_empty_batch() {
        let b = be(vec![]).unsqueeze(0);
        assert_eq!(vec![1], *b.batch_shape());
    }

    #[test]
    fn unsqueeze_last() {
        let b = be(vec![2, 3]).unsqueeze(-1);
        assert_eq!(vec![2, 3, 1], *b.batch_shape());
    }

    #[test]
    fn unsqueeze_neg_idx() {
        let b = be(vec![2, 5, 2]).unsqueeze(-2);
        assert_eq!(vec![2, 5, 1, 2], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn unsqueeze_panic() {
        be(vec![2, 3]).unsqueeze(3);
    }

    // ───────────── stack ─────────────

    #[test]
    fn stack() {
        let b = BatchEmpty::stack(vec![be(vec![2, 3]), be(vec![2, 3]), be(vec![2, 3])]);
        assert_eq!(vec![3, 2, 3], *b.batch_shape());
    }

    #[test]
    fn stack_empty_batch() {
        // Single 환경 원소([]) N개 → [N]: sync wrapper가 쓰는 경로
        let b = BatchEmpty::stack(vec![BatchEmpty::<()>::with_empty_batch_dim(), BatchEmpty::with_empty_batch_dim(), BatchEmpty::with_empty_batch_dim()]);
        assert_eq!(vec![3], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn stack_mismatch_panic() {
        BatchEmpty::stack(vec![be(vec![2]), be(vec![3])]);
    }

    #[test]
    #[should_panic]
    fn stack_empty_vec_panic() {
        BatchEmpty::<()>::stack(vec![]);
    }

    // ───────────── cat ─────────────

    #[test]
    fn cat_dim0() {
        let b = BatchEmpty::cat(vec![be(vec![2, 3]), be(vec![4, 3])], 0);
        assert_eq!(vec![6, 3], *b.batch_shape());
    }

    #[test]
    fn cat_dim1() {
        let b = BatchEmpty::cat(vec![be(vec![3, 1, 2]), be(vec![3, 4, 2])], 1);
        assert_eq!(vec![3, 5, 2], *b.batch_shape());
    }

    #[test]
    fn cat_neg_dim() {
        let b = BatchEmpty::cat(vec![be(vec![2, 1]), be(vec![2, 2])], -1);
        assert_eq!(vec![2, 3], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn cat_mismatch_panic() {
        BatchEmpty::cat(vec![be(vec![2, 1]), be(vec![3, 2])], 1);
    }

    #[test]
    #[should_panic]
    fn cat_rank_mismatch_panic() {
        BatchEmpty::cat(vec![be(vec![2, 1]), be(vec![2])], 0);
    }

    #[test]
    #[should_panic]
    fn cat_empty_vec_panic() {
        BatchEmpty::<()>::cat(vec![], 0);
    }

    #[test]
    #[should_panic]
    fn cat_on_empty_batch_panic() {
        // 배치 [] 에는 이어 붙일 축이 없음
        BatchEmpty::cat(vec![be(vec![]), be(vec![])], 0);
    }

    // ───────────── slice ─────────────

    #[test]
    fn slice_dim0() {
        let b = be(vec![4]).slice(0, (1..3).into());
        assert_eq!(vec![2], *b.batch_shape());
    }

    #[test]
    fn slice_dim1() {
        let b = be(vec![2, 3]).slice(1, (1..3).into());
        assert_eq!(vec![2, 2], *b.batch_shape());
    }

    #[test]
    fn slice_neg_dim() {
        let b = be(vec![2, 3]).slice(-1, (0..1).into());
        assert_eq!(vec![2, 1], *b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn slice_out_of_range_panic() {
        // BatchTensor와 같이 범위를 벗어나면 패닉해야 함
        be(vec![2, 3]).slice(1, (2..5).into());
    }

    #[test]
    #[should_panic]
    fn slice_dim_out_of_range_panic() {
        be(vec![2, 3]).slice(2, (0..1).into());
    }

    // ───────────── select ─────────────

    #[test]
    fn select_dim0() {
        let b = be(vec![2, 3]).select(0, Tensor::from_ints([1], &d()));
        assert_eq!(vec![1, 3], *b.batch_shape());
    }

    #[test]
    fn select_dim1_with_duplicates() {
        let b = be(vec![2, 3]).select(1, Tensor::from_ints([2, 0, 2], &d()));
        assert_eq!(vec![2, 3], *b.batch_shape());
    }

    #[test]
    fn select_neg_dim() {
        let b = be(vec![2, 3]).select(-1, Tensor::from_ints([1], &d()));
        assert_eq!(vec![2, 1], *b.batch_shape());
    }

    #[test]
    fn select_after_flatten() {
        let b = be(vec![2, 3]).reshape(vec![6]).select(0, Tensor::from_ints([5, 0], &d()));
        assert_eq!(vec![2], *b.batch_shape());
    }

    // ───────────── zeros_like / assign_inplace ─────────────

    #[test]
    fn zeros_like() {
        let z = BatchEmpty::zeros_like(4, &be(vec![3]));
        assert_eq!(vec![4, 3], *z.batch_shape());
    }

    #[test]
    fn zeros_like_from_empty_batch() {
        let z = BatchEmpty::zeros_like(4, &BatchEmpty::<()>::with_empty_batch_dim());
        assert_eq!(vec![4], *z.batch_shape());
    }

    #[test]
    fn assign_inplace_keeps_shape() {
        let mut z = BatchEmpty::zeros_like(4, &be(vec![2]));
        z.assign_inplace(be(vec![1, 2]), 2);
        assert_eq!(vec![4, 2], *z.batch_shape());
    }

    #[test]
    #[should_panic]
    fn assign_inplace_out_of_range_panic() {
        // BatchTensor와 같이 범위를 벗어나면 패닉해야 함
        let mut z = BatchEmpty::zeros_like(4, &be(vec![2]));
        z.assign_inplace(be(vec![2, 2]), 3);
    }

    #[test]
    #[should_panic]
    fn assign_inplace_inner_mismatch_panic() {
        // BatchTensor와 같이 0번 축 외의 배치가 다르면 패닉해야 함
        let mut z = BatchEmpty::zeros_like(4, &be(vec![2]));
        z.assign_inplace(be(vec![1, 3]), 0);
    }
}