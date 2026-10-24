use ndarray::prelude::*;

use crate::core::data::{WithBatchDim, batch_dim::{BatchSelect, BatchShape, ops::*}};

pub struct BatchArray<A> {
    nb: usize,
    array: ArrayD<A>
}

impl<A: Default> WithBatchDim for BatchArray<A> {
    fn batch_shape(&self) -> &BatchShape {
        self.array.shape()[..self.nb].try_into().unwrap()
    }

    fn reshape(self, shape: BatchShape) -> Self {
        assert!(check_reshape(self.batch_shape(), &shape), "invalid shape was given for reshape");
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
        let (shape, _) = check_squeeze(self.shape, dim).unwrap();
        self.reshape(shape)
    }

    fn unsqueeze(self, dim: isize) -> Self {
        let (shape, _) = check_unsqueeze(self.shape, dim).unwrap();
        self.reshape(shape)
    }

    fn stack(dat: Vec<Self>) -> Self {
        let shape = check_stack(&dat).unwrap();
        let inner = dat.into_iter().map(|a| a.array).collect();
        let stacked = ndarray::stack(Axis(0), inner).unwrap();
        Self {
            nb: shape.len(),
            array: stacked
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        let (shape, dim) = check_cat(dim, &dat).unwrap();
        let inner = dat.into_iter().map(|a| a.array).collect();
        let catenated = ndarray::concatenate(Axis(dim), inner).unwrap();
        Self {
            nb: shape.len(),
            array: catenated
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let (shape, dim) = check_slice(self.batch_shape().clone(), dim, range).unwrap();
        let a = self.array.slice_axis(Axis(dim), ndarray::Slice::from(range)).to_owned();
        Self {
            nb: shape.len(),
            array: a
        }
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        check_assign_inplace(self.batch_shape(), data.batch_shape(), idx).unwrap();
        let k = data.array.shape()[0];
        self.array.slice_axis_mut(Axis(0), ndarray::Slice::from(idx..idx + k)).assign(&data.array);
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let dims = data.array.shape();
        let new_dims = vec![capacity];
        new_dims.extend_from_slice(dims);
        let a = ArrayD::from_elem(dims, A::default());
        Self {
            nb: data.nb + 1,
            array: a
        }
    }
}

impl<A: Default> BatchSelect<Vec<usize>> for BatchArray<A> {
    fn select(self, dim: isize, idx: Vec<usize>) -> Self {
        let (shape, dim) = check_select(self.shape, dim, idx).unwrap();
        let a = self.array.select(Axis(dim), &idx);
        Self {
            nb: shape.len(),
            array: a
        }
    }
}