use ndarray::prelude::*;
use burn::prelude::*;
use crate::core::data::{WithBatchDim, batch_dim::{BatchShape, convert::{AttachBatchDim, DetachBatchDim}, ops::*}};

/// ndarray::ArrayD with batch dimension
#[derive(Debug, Clone)]
pub struct BatchArray<A: Default + Clone + 'static> {
    nb: usize,
    array: ArrayD<A>
}

impl<A: Default + Clone + 'static> BatchArray<A> {
    /// create a new `BatchArray` with empty batch dimension, from given tensor
    pub fn with_empty_batch_dim(array: ArrayD<A>) -> Self {
        Self {
            nb: 0,
            array
        }
    }
    /// create a new `BatchArray` with given batch dimension
    pub fn with_batch_dim(batch_shape: BatchShape, array: ArrayD<A>) -> Self {
        assert!(batch_shape == array.shape()[..batch_shape.len()], "invalid batch shape given for with_batch_dim of BatchTensor");

        Self {
            nb: batch_shape.len(),
            array
        }
    }
}

impl<A: Default + Send + Sync + Clone + 'static> WithBatchDim for BatchArray<A> {
    fn batch_shape(&self) -> BatchShape {
        self.array.shape()[..self.nb].iter().map(|v| *v).collect()
    }

    fn reshape(self, shape: BatchShape) -> Self {
        assert!(check_reshape(&self.batch_shape(), &shape), "invalid shape was given for reshape");
        let new_nb = shape.len();
        let mut dims = shape;
        dims.extend_from_slice(&self.array.shape()[self.nb..]);
        let a = self.array.as_standard_layout().into_owned().into_shape_with_order(IxDyn(&dims)).unwrap();
        Self {
            nb: new_nb,
            array: a
        }
    }

    fn squeeze(self, dim: isize) -> Self {
        let (shape, _) = check_squeeze(self.batch_shape(), dim).unwrap();
        self.reshape(shape)
    }

    fn unsqueeze(self, dim: isize) -> Self {
        let (shape, _) = check_unsqueeze(self.batch_shape(), dim).unwrap();
        self.reshape(shape)
    }

    fn stack(dat: Vec<Self>) -> Self {
        let shape = check_stack(&dat).unwrap();
        let inner: Vec<_> = dat.iter().map(|a| a.array.view()).collect();
        let stacked = ndarray::stack(Axis(0), &inner).unwrap();
        Self {
            nb: shape.len(),
            array: stacked
        }
    }

    fn unstack(self) -> Vec<Self> {
        assert!(self.nb != 0, "Given BatchArray does not have any batch dimension");

        let items = self.array.axis_iter(Axis(0))
        .map(|v| BatchArray { nb: self.nb - 1, array: v.to_owned() })
        .collect();

        items
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        let (shape, dim) = check_cat(dim, &dat).unwrap();
        let inner: Vec<_> = dat.iter().map(|a| a.array.view()).collect();
        let catenated = ndarray::concatenate(Axis(dim), &inner).unwrap();
        Self {
            nb: shape.len(),
            array: catenated
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let (shape, dim) = check_slice(self.batch_shape().clone(), dim, range).unwrap();
        let a = self.array.slice_axis(Axis(dim), ndarray::Slice::from(range.start..range.end)).to_owned();
        Self {
            nb: shape.len(),
            array: a
        }
    }

    fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let (shape, dim) = check_select(self.array.shape()[..self.nb].to_vec(), dim, idx.clone()).unwrap();
        let idx: Vec<usize> = idx.try_into_vec_as::<u64>().unwrap().into_iter().map(|u| u as usize).collect();
        let a = self.array.select(Axis(dim), &idx);
        Self {
            nb: shape.len(),
            array: a
        }
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        check_assign_inplace(&self.batch_shape(), &data.batch_shape(), idx).unwrap();
        let k = data.array.shape()[0];
        self.array.slice_axis_mut(Axis(0), ndarray::Slice::from(idx..idx + k)).assign(&data.array);
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let dims = data.array.shape();
        let mut new_dims = vec![capacity];
        new_dims.extend_from_slice(dims);
        let a = ArrayD::from_elem(IxDyn(&new_dims), A::default());
        Self {
            nb: data.nb + 1,
            array: a
        }
    }
}

impl<A: Default + Send + Sync + Clone + 'static> AttachBatchDim for ArrayD<A> {
    type Attached = BatchArray<A>;

    fn attach_batch_dim(self, batch_shape: BatchShape) -> Self::Attached {
        BatchArray::with_batch_dim(batch_shape, self)
    }
}

impl<A: Default + Send + Sync + Clone + 'static> DetachBatchDim for BatchArray<A> {
    type Detached = ArrayD<A>;

    fn detach_batch_dim(self) -> (Self::Detached, BatchShape) {
        let shape = self.array.shape()[..self.nb].try_into().unwrap();
        (self.array, shape)
    }
}

// tests made by claude

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use crate::core::data::{BatchShape, WithBatchDim};
    use crate::core::data::batch_dim::{convert::{AttachBatchDim, DetachBatchDim}};
    use super::BatchArray;

    /// 배치 원소 i의 특징이 [i, i + 0.5]인 배열 (실제 축: [batch…, 2])
    fn seq(shape: BatchShape) -> BatchArray<f32> {
        let n: usize = shape.iter().product();
        let mut dims = shape.clone();
        dims.push(2);
        let v: Vec<f32> = (0..n).flat_map(|i| [i as f32, i as f32 + 0.5]).collect();
        ArrayD::from_shape_vec(IxDyn(&dims), v).unwrap().attach_batch_dim(shape)
    }

    /// 각 배치 원소의 원래 번호 (특징 행이 깨지지 않았는지도 확인)
    fn ids(b: &BatchArray<f32>) -> Vec<usize> {
        let (a, shape) = b.clone().detach_batch_dim();
        let n: usize = shape.iter().product();
        let flat = a.as_standard_layout().into_owned().into_shape_with_order((n, 2)).unwrap();
        flat.rows().into_iter().map(|r| { assert_eq!(r[1], r[0] + 0.5); r[0] as usize }).collect()
    }

    // ───────────── attach / detach ─────────────

    #[test]
    fn attach_detach_roundtrip() {
        let a = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4]));
        let b = a.clone().attach_batch_dim(vec![2, 3]);
        assert_eq!(vec![2, 3], b.batch_shape());
        let (back, shape) = b.detach_batch_dim();
        assert_eq!(back, a);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn attach_empty_batch() {
        // 실제 축 방식: 배치 [] 이면 배치 축이 없음 (특징 축은 있어도 됨)
        let b = ArrayD::<f32>::zeros(IxDyn(&[4])).attach_batch_dim(vec![]);
        assert!(b.batch_shape().is_empty());
    }

    #[test]
    fn with_empty_batch_dim_accepts_feature_array() {
        // 배치 [] 원소는 [F] 모양이어야 함
        let b = BatchArray::with_empty_batch_dim(ArrayD::<f32>::zeros(IxDyn(&[4])));
        assert!(b.batch_shape().is_empty());
    }

    #[test]
    #[should_panic]
    fn attach_mismatch_panic() {
        ArrayD::<f32>::zeros(IxDyn(&[2, 3])).attach_batch_dim(vec![3]);
    }

    // ───────────── 메타데이터 ─────────────

    #[test]
    fn reshape_keeps_values() {
        let b = seq(vec![2, 3]).reshape(vec![6]);
        assert_eq!(vec![6], b.batch_shape());
        assert_eq!(vec![0, 1, 2, 3, 4, 5], ids(&b));
    }

    #[test]
    #[should_panic]
    fn reshape_panic() {
        seq(vec![2, 3]).reshape(vec![4]);
    }

    #[test]
    fn squeeze_unsqueeze() {
        let b = seq(vec![2, 1, 3]).squeeze(-2);
        assert_eq!(vec![2, 3], b.batch_shape());
        let b = b.unsqueeze(-1);
        assert_eq!(vec![2, 3, 1], b.batch_shape());
        assert_eq!(vec![0, 1, 2, 3, 4, 5], ids(&b));
    }

    #[test]
    fn unsqueeze_empty_batch() {
        let b = ArrayD::<f32>::zeros(IxDyn(&[4])).attach_batch_dim(vec![]).unsqueeze(0);
        assert_eq!(vec![1], b.batch_shape());
    }

    #[test]
    #[should_panic]
    fn squeeze_not_1_panic() {
        seq(vec![2, 3]).squeeze(0);
    }

    // ───────────── stack / unstack / cat ─────────────

    #[test]
    fn stack_values() {
        let b = BatchArray::stack(vec![seq(vec![2]), seq(vec![2]), seq(vec![2])]);
        assert_eq!(vec![3, 2], b.batch_shape());
        assert_eq!(vec![0, 1, 0, 1, 0, 1], ids(&b));
    }

    #[test]
    fn stack_empty_batch() {
        // Single 환경 원소([]) N개 → [N]
        let one = || ArrayD::<f32>::zeros(IxDyn(&[4])).attach_batch_dim(vec![]);
        let b = BatchArray::stack(vec![one(), one(), one()]);
        assert_eq!(vec![3], b.batch_shape());
        assert_eq!(b.detach_batch_dim().0.shape(), &[3, 4]);
    }

    #[test]
    #[should_panic]
    fn stack_mismatch_panic() {
        BatchArray::stack(vec![seq(vec![2]), seq(vec![3])]);
    }

    #[test]
    fn unstack_values() {
        let parts = seq(vec![2, 3]).unstack();
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|p| p.batch_shape() == vec![3]));
        assert_eq!(ids(&parts[0]), vec![0, 1, 2]);
        assert_eq!(ids(&parts[1]), vec![3, 4, 5]);
    }

    #[test]
    fn stack_unstack_roundtrip() {
        let parts = vec![seq(vec![2]), seq(vec![2]), seq(vec![2])];
        let back = BatchArray::stack(parts).unstack();
        assert_eq!(back.len(), 3);
        assert!(back.iter().all(|p| ids(p) == vec![0, 1]));
    }

    #[test]
    #[should_panic]
    fn unstack_empty_batch_panic() {
        ArrayD::<f32>::zeros(IxDyn(&[4])).attach_batch_dim(vec![]).unstack();
    }

    #[test]
    fn cat_dim1_values() {
        let b = BatchArray::cat(vec![seq(vec![2, 1]), seq(vec![2, 2])], 1);
        assert_eq!(vec![2, 3], b.batch_shape());
        assert_eq!(vec![0, 0, 1, 1, 2, 3], ids(&b));
    }

    #[test]
    #[should_panic]
    fn cat_mismatch_panic() {
        BatchArray::cat(vec![seq(vec![2, 1]), seq(vec![3, 2])], 1);
    }

    // ───────────── slice / select ─────────────

    #[test]
    fn slice_dim1_values() {
        let b = seq(vec![2, 3]).slice(1, (1..3).into());
        assert_eq!(vec![2, 2], b.batch_shape());
        assert_eq!(vec![1, 2, 4, 5], ids(&b));
    }

    #[test]
    #[should_panic]
    fn slice_out_of_range_panic() {
        seq(vec![2, 3]).slice(1, (2..5).into());
    }

    // #[test]
    // fn select_dim1_values() {
    //     let b = seq(vec![2, 3]).select(1, Tensor::from_ints([2, 0, 2]));
    //     assert_eq!(vec![2, 3], b.batch_shape());
    //     assert_eq!(vec![2, 0, 2, 5, 3, 5], ids(&b));
    // }

    // #[test]
    // fn select_after_flatten() {
    //     let b = seq(vec![2, 3]).reshape(vec![6]).select(0, vec![5, 0]);
    //     assert_eq!(vec![5, 0], ids(&b));
    // }

    // ───────────── zeros_like / assign_inplace ─────────────

    #[test]
    fn zeros_like_shape() {
        let z = BatchArray::zeros_like(4, &seq(vec![3]));
        assert_eq!(vec![4, 3], z.batch_shape());
        let (a, _) = z.detach_batch_dim();
        assert_eq!(a.shape(), &[4, 3, 2]);
        assert!(a.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn assign_inplace_values() {
        let mut z = BatchArray::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![2]).unsqueeze(0), 2);
        let (a, _) = z.detach_batch_dim();
        assert_eq!(a.index_axis(Axis(0), 2).iter().cloned().collect::<Vec<f32>>(), vec![0.0, 0.5, 1.0, 1.5]);
        assert!(a.index_axis(Axis(0), 0).iter().all(|x| *x == 0.0));
    }

    #[test]
    #[should_panic]
    fn assign_inplace_out_of_range_panic() {
        let mut z = BatchArray::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![2, 2]), 3);
    }
}