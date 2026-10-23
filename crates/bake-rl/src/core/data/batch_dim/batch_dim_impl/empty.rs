use std::marker::PhantomData;

use crate::core::data::batch_dim::{BatchShape, WithBatchDim};
use crate::core::data::batch_dim::ops::*;

/// `Unconstrained` or any empty struct with batch dimension
/// - I may change return type of `batch_size()` into `Option` and delete this struct
#[derive(Debug, Clone)]
pub struct BatchEmpty<T> {
    shape: BatchShape,
    _p: PhantomData<fn() -> T>
}

impl<T> BatchEmpty<T> {
    /// create a new `BatchEmpty<T>` struct with empty batch dimension
    pub fn new() -> Self {
        Self {
            shape: vec![],
            _p: PhantomData
        }
    }
}

impl<T> WithBatchDim for BatchEmpty<T> {
    fn batch_shape(&self) -> &BatchShape {
        &self.shape
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

    fn assign_inplace(&mut self, _: Self, _: usize) {}

    fn stack(dat: Vec<Self>) -> Self {
        let shape = check_stack(&dat).unwrap();
        Self {
            shape,
            _p: PhantomData
        }
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

    fn select(self, dim: isize, idx: burn::prelude::Tensor<1, burn::prelude::Int>) -> Self {
        let (shape, _) = check_select(self.shape, dim, idx).unwrap();
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