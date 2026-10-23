use std::marker::PhantomData;

use crate::core::data::batch_dim::{BatchShape, WithBatchDim, check_batch_dim, check_batch_dim_except, resolve_negative_index};

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
        let m = self.shape.into_iter().product::<usize>();
        let n = shape.iter().product::<usize>();

        if m != n { panic!("Invalid shape was given for reshape operation") }

        Self { shape, _p: PhantomData }
    }

    fn squeeze(mut self, dim: isize) -> Self {
        let dim = resolve_negative_index(self.shape.len(), dim);

        if self.shape[dim] != 1 { panic!("The dimension is not 1") }
        self.shape.remove(dim);
        self
    }

    fn unsqueeze(mut self, dim: isize) -> Self {
        let dim = resolve_negative_index(self.shape.len() + 1, dim);

        self.shape.insert(dim, 1);

        self
    }

    fn assign_inplace(&mut self, _: Self, _: usize) {}

    fn stack(dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "Given data vector is empty");
        assert!(check_batch_dim(&dat), "The batch dimension does not matches");

        let n = dat.len();
        let mut shape = dat[0].shape.clone();
        shape.insert(0, n);
        
        Self {
            shape,
            _p: PhantomData
        }
    }
    
    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        assert!(!dat.is_empty(), "The given data is empty");

        let mut shape = dat[0].shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);
        assert!(check_batch_dim_except(&dat, dim), "The batch dimension does not matches");

        let mut k_final = 0;
        for d in dat.into_iter() {
            k_final += d.shape[dim];
        }
        shape[dim] = k_final;

        Self {
            shape,
            _p: PhantomData
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);

        let k_final = range.iter().len();
        shape[dim] = k_final;

        Self {
            shape,
            _p: PhantomData
        }
    }

    fn select(self, dim: isize, idx: burn::prelude::Tensor<1, burn::prelude::Int>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);

        let k_final = idx.dims()[0];
        shape[dim] = k_final;
        
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