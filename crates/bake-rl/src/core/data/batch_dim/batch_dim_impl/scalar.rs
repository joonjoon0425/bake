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
                    One(a.detach_batch_dim().0[0])
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
                    unstacked.into_iter().map(|a| One(a.detach_batch_dim().0[0])).collect()
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
            One(k) => { 
                Many(ArrayD::from_elem(vec![capacity], *k).attach_batch_dim(vec![capacity]))
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