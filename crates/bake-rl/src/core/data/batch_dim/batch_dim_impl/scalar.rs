use crate::core::{ElementKind, data::{BatchArray, WithBatchDim, batch_dim::{BatchSelect, convert::{AttachBatchDim, DetachBatchDim}, ops::{check_reshape, check_unsqueeze}}}};
use ndarray::prelude::*;

/// Batchable Scalar type
#[derive(Debug, Clone)]
pub enum BatchScalar<K: ElementKind> {
    /// contains only one element
    One(K::Kind),
    /// contains scalar as batches
    Many(BatchArray<K::Kind>)
}

impl<K: ElementKind> BatchScalar<K> {
    /// extract the scalar
    /// # Panics
    /// - If `Many` is the current enumeration, panics
    pub fn into_scalar(self) -> K::Kind {
        match self {
            One(k) => k,
            Many(_) => panic!("BatchScalar::into_scalar: not One(_)")
        }
    }
}

use BatchScalar::One;
use BatchScalar::Many;

impl<K: ElementKind> WithBatchDim for BatchScalar<K> {
    fn batch_shape(&self) -> crate::core::data::BatchShape {
        match self {
            One(_) => vec![],
            Many(a) => a.batch_shape()
        }
    }

    fn reshape(self, shape: crate::core::data::BatchShape) -> Self {
        match self {
            One(k) => {
                assert!(shape.is_empty(), "reshape: batch shape is not empty");
                One(k)
            },
            Many(a) => {
                if shape.is_empty() && check_reshape(&a.batch_shape(), &shape) {
                    One(a.detach_batch_dim().0.iter().next().cloned().unwrap())
                } else {
                    Many(a.reshape(shape))
                }
            }
        }
    }

    fn squeeze(self, dim: isize) -> Self {
        match self {
            One(_) => {
                panic!("squeeze: cannot squeeze pure scalar");
            },
            Many(a) => Many(a.squeeze(dim))
        }
    }

    fn unsqueeze(self, dim: isize) -> Self {
        match self {
            One(k) => {
                let (shape, _) = check_unsqueeze(vec![], dim).unwrap();
                let a = ArrayD::from_elem(vec![1], k);
                Many(a.attach_batch_dim(shape))
            },
            Many(a) => {
                Many(a.unsqueeze(dim))
            }
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        if dat.iter().all(|x| matches!(x, BatchScalar::Many(_))) {
            let v: Vec<_> = dat.into_iter()
                .map(|x| match x { One(_) => unreachable!(), Many(a) => a })
                .collect();
            let a = WithBatchDim::cat(v, dim);
            BatchScalar::Many(a)
        } else {
            panic!("scalar values cannot be catenated; use stack instead");
        }
    }

    fn stack(dat: Vec<Self>) -> Self {
        if dat.iter().all(|x| matches!(x, BatchScalar::One(_))) {
            let n = dat.len();
            let v: Vec<K::Kind> = dat.into_iter()
                .map(|x| match x { BatchScalar::One(v) => v, BatchScalar::Many(_) => unreachable!() })
                .collect();
            let a = ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap();
            BatchScalar::Many(a.attach_batch_dim(vec![n]))
        } else {
            let v: Vec<_> = dat.into_iter()
                .map(|x| match x { One(_) => unreachable!(), Many(a) => a })
                .collect();
            let a = WithBatchDim::stack(v);
            BatchScalar::Many(a)
        }
    }

    fn unstack(self) -> Vec<Self> {
        match self {
            One(_) => panic!("scalar values cannot be unstacked"),
            Many(a) => {
                let unstacked = a.unstack();
                if unstacked.iter().all(|a| a.batch_shape().is_empty()) {
                    unstacked.into_iter().map(|a| One(a.detach_batch_dim().0.iter().next().cloned().unwrap())).collect()
                } else {
                    unstacked.into_iter().map(|a| Many(a)).collect()
                }
            }
        }
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        match (self, data) {
            (One(d), One(s)) => {
                *d = s;
            },
            (Many(d), Many(s)) => {
                d.assign_inplace(s, idx);
            },
            (_, _) => panic!("BatchScalar::assign_inplace: data and self has differrent enumeration type"),
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        match self {
            One(_) => panic!("BatchScalar::slice: cannot slice scalar"),
            Many(a) => {
                Many(a.slice(dim, range))
            }
        }
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        match data {
            One(_) => { 
                Many(ArrayD::from_elem(vec![capacity], K::Kind::default()).attach_batch_dim(vec![capacity]))
            },
            Many(a) => {
                Many(WithBatchDim::zeros_like(capacity, a))
            }
        }
    }
}

impl<K: ElementKind> BatchSelect<Vec<usize>> for BatchScalar<K> {
    fn select(self, dim: isize, idx: Vec<usize>) -> Self {
        match self {
            One(_) => panic!("BatchScalar::select: cannot call select on scalar values"),
            Many(a) => Many(a.select(dim, idx))
        }
    }
}

impl<K: ElementKind> DetachBatchDim for BatchScalar<K> {
    type Detached = Self;
    fn detach_batch_dim(self) -> (Self::Detached, crate::core::data::BatchShape) {
        let shape = self.batch_shape();
        (self, shape)
    }
}

// tests made by Claude

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use ndarray::prelude::*;
    use crate::core::data::{BatchScalar, WithBatchDim};
    use crate::core::data::batch_dim::{BatchSelect, convert::AttachBatchDim};

    fn many(shape: Vec<usize>) -> BatchScalar<Int> {
        // 원소 값 = 평평한 위치 번호
        let n: usize = shape.iter().product();
        let a = ArrayD::from_shape_vec(IxDyn(&shape), (0..n as i64).collect()).unwrap();
        BatchScalar::Many(a.attach_batch_dim(shape))
    }
    fn values(s: &BatchScalar<Int>) -> Vec<i64> {
        match s {
            BatchScalar::One(v) => vec![*v],
            BatchScalar::Many(a) => {
                let (arr, _) = crate::core::data::batch_dim::convert::DetachBatchDim::detach_batch_dim(a.clone());
                arr.as_standard_layout().iter().cloned().collect()
            }
        }
    }

    // ───────────── One ─────────────

    #[test]
    fn one_is_empty_batch() {
        assert!(BatchScalar::<Int>::One(3).batch_shape().is_empty());
        assert_eq!(BatchScalar::<Int>::One(3).into_scalar(), 3);
    }

    #[test]
    #[should_panic]
    fn into_scalar_many_panic() {
        many(vec![2]).into_scalar();
    }

    #[test]
    fn unsqueeze_one() {
        let s = BatchScalar::<Int>::One(7).unsqueeze(0);
        assert_eq!(vec![1], s.batch_shape());
        assert_eq!(values(&s), vec![7]);
    }

    #[test]
    #[should_panic]
    fn squeeze_one_panic() {
        BatchScalar::<Int>::One(7).squeeze(0);
    }

    // ───────────── stack / unstack (SyncEnv 경로) ─────────────

    #[test]
    fn stack_ones() {
        let s = BatchScalar::stack(vec![BatchScalar::<Int>::One(1), BatchScalar::One(2), BatchScalar::One(3)]);
        assert_eq!(vec![3], s.batch_shape());
        assert_eq!(values(&s), vec![1, 2, 3]);
    }

    #[test]
    fn unstack_to_ones() {
        // [3] → One 3개
        let parts = many(vec![3]).unstack();
        assert_eq!(parts.len(), 3);
        let v: Vec<i64> = parts.into_iter().map(|p| p.into_scalar()).collect();
        assert_eq!(v, vec![0, 1, 2]);
    }

    #[test]
    fn unstack_to_many() {
        let parts = many(vec![2, 3]).unstack();
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|p| p.batch_shape() == vec![3]));
        assert_eq!(values(&parts[1]), vec![3, 4, 5]);
    }

    #[test]
    fn stack_unstack_roundtrip() {
        let ones: Vec<_> = (0..4).map(|i| BatchScalar::<Int>::One(i)).collect();
        let back = BatchScalar::stack(ones).unstack();
        assert_eq!(back.into_iter().map(|p| p.into_scalar()).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn stack_manys() {
        let s = BatchScalar::stack(vec![many(vec![2]), many(vec![2])]);
        assert_eq!(vec![2, 2], s.batch_shape());
        assert_eq!(values(&s), vec![0, 1, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn unstack_one_panic() {
        BatchScalar::<Int>::One(1).unstack();
    }

    // ───────────── reshape ─────────────

    #[test]
    fn reshape_many_to_empty_becomes_one() {
        // [1] → [] 이면 One으로 돌아와야 함
        let s = BatchScalar::<Int>::One(5).unsqueeze(0).reshape(vec![]);
        assert!(matches!(s, BatchScalar::One(5)));
    }

    #[test]
    fn reshape_many_2d_to_empty_becomes_one() {
        // [1, 1] → []
        let s = BatchScalar::<Int>::One(5).unsqueeze(0).unsqueeze(0).reshape(vec![]);
        assert!(matches!(s, BatchScalar::One(5)));
    }

    #[test]
    fn reshape_many() {
        let s = many(vec![2, 3]).reshape(vec![6]);
        assert_eq!(vec![6], s.batch_shape());
        assert_eq!(values(&s), vec![0, 1, 2, 3, 4, 5]);
    }

    // ───────────── cat / slice / select ─────────────

    #[test]
    fn cat_manys() {
        let s = BatchScalar::cat(vec![many(vec![2]), many(vec![3])], 0);
        assert_eq!(vec![5], s.batch_shape());
        assert_eq!(values(&s), vec![0, 1, 0, 1, 2]);
    }

    #[test]
    #[should_panic]
    fn cat_ones_panic() {
        BatchScalar::cat(vec![BatchScalar::<Int>::One(1), BatchScalar::One(2)], 0);
    }

    #[test]
    fn slice_and_select() {
        let s = many(vec![2, 3]).slice(1, (1..3).into());
        assert_eq!(values(&s), vec![1, 2, 4, 5]);
        let s = many(vec![2, 3]).select(1, vec![2, 0]);
        assert_eq!(values(&s), vec![2, 0, 5, 3]);
    }

    // ───────────── zeros_like / assign_inplace ─────────────

    #[test]
    fn zeros_like_from_one_is_zero() {
        // One(7)로부터 만든 버퍼도 0(기본값)으로 채워져야 함
        let z = BatchScalar::zeros_like(4, &BatchScalar::<Int>::One(7));
        assert_eq!(vec![4], z.batch_shape());
        assert_eq!(values(&z), vec![0, 0, 0, 0]);
    }

    #[test]
    fn zeros_like_from_many() {
        let z = BatchScalar::zeros_like(2, &many(vec![3]));
        assert_eq!(vec![2, 3], z.batch_shape());
        assert_eq!(values(&z), vec![0; 6]);
    }

    #[test]
    fn assign_step_into_buffer() {
        // 버퍼 [4]에 스텝 하나(One)를 unsqueeze해서 쓰기
        let mut buf = BatchScalar::zeros_like(4, &BatchScalar::<Int>::One(0));
        buf.assign_inplace(BatchScalar::One(9).unsqueeze(0), 2);
        assert_eq!(values(&buf), vec![0, 0, 9, 0]);
    }
}