use ndarray::prelude::*;

use crate::core::data::{WithBatchDim, batch_dim::{BatchSelect, BatchShape, convert::{AttachBatchDim, DetachBatchDim}, ops::*}};

pub struct BatchArray<A: Default + Clone + 'static> {
    nb: usize,
    array: ArrayD<A>
}

impl<A: Default + Clone + 'static> BatchArray<A> {
    /// create a new `BatchArray` with empty batch dimension, from given tensor
    pub fn with_empty_batch_dim(array: ArrayD<A>) -> Self {
        assert_eq!(array.shape()[0], 1, "empty batch shape requires dim 0 == 1");

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

impl<A: Default + Clone + 'static> WithBatchDim for BatchArray<A> {
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
        let inner = dat.into_iter().map(|a| a.array.view()).collect();
        let stacked = ndarray::stack(Axis(0), inner).unwrap();
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
        let inner = dat.iter().map(|a| a.array.view()).collect();
        let catenated = ndarray::concatenate(Axis(dim), inner).unwrap();
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

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        check_assign_inplace(&self.batch_shape(), &data.batch_shape(), idx).unwrap();
        let k = data.array.shape()[0];
        self.array.slice_axis_mut(Axis(0), ndarray::Slice::from(idx..idx + k)).assign(&data.array);
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let dims = data.array.shape();
        let mut new_dims = vec![capacity];
        new_dims.extend_from_slice(dims);
        let a = ArrayD::from_elem(dims, A::default());
        Self {
            nb: data.nb + 1,
            array: a
        }
    }
}

impl<A: Default + Clone + 'static> BatchSelect<Vec<usize>> for BatchArray<A> {
    fn select(self, dim: isize, idx: Vec<usize>) -> Self {
        let (shape, dim) = check_select(self.array.shape()[..self.nb].to_vec(), dim, idx.clone()).unwrap();
        let a = self.array.select(Axis(dim), &idx);
        Self {
            nb: shape.len(),
            array: a
        }
    }
}

impl<A: Default + Clone + 'static> AttachBatchDim for ArrayD<A> {
    type Attached = BatchArray<A>;

    fn attach_batch_dim(self, batch_shape: BatchShape) -> Self::Attached {
        BatchArray::with_batch_dim(batch_shape, self)
    }
}

impl<A: Default + Clone + 'static> DetachBatchDim for BatchArray<A> {
    type Detached = ArrayD<A>;

    fn detach_batch_dim(self) -> (Self::Detached, BatchShape) {
        let shape = self.array.shape()[..self.nb].try_into().unwrap();
        (self.array, shape)
    }
}