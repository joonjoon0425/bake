use burn::{prelude::*, tensor::kind::Basic};

use crate::core::data::batch_dim::{BatchShape, WithBatchDim, check_batch_dim, check_batch_dim_except, n_batch, resolve_negative_index};

/// `burn::Tensor`, with batch dimension
pub struct BatchTensor<const D: usize, K: Basic = Float> {
    shape: BatchShape,
    t: Tensor<D, K>,
}

impl<const D: usize, K: Basic> BatchTensor<D, K> {
    /// create a new `BatchTensor` with empty batch dimension, from given tensor
    pub fn new(tensor: Tensor<D, K>) -> Self {
        Self {
            shape: vec![],
            t: tensor
        }
    }
}

impl<const D: usize, K: Basic> WithBatchDim for BatchTensor<D, K> {
    fn batch_shape(&self) -> &BatchShape {
        &self.shape
    }

    fn reshape(self, shape: BatchShape) -> Self {
        let m = self.shape.into_iter().product::<usize>();
        let n = shape.iter().product::<usize>();

        if m != n { panic!("Invalid shape was given for reshape operation") }

        Self {
            shape,
            ..self
        }
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

    fn assign_inplace(&mut self, data: Self, idx: usize) {

        let row = n_batch(&self.shape[1..]);
        let k = n_batch(&data.shape);
        self.t.inplace(|a| a.slice_assign((idx * row)..(idx + k) * row, data.t));
    }

    fn stack(dat: Vec<Self>) -> Self {
        assert!(dat.is_empty(), "Given data vector is empty");
        assert!(check_batch_dim(&dat), "The batch dimension does not matches");

        let n = dat.len();
        let mut shape = dat[0].shape.clone();
        shape.insert(0, n);
        let inner = dat.into_iter().map(|d| d.t).collect();
        let stacked = Tensor::stack(inner, 0);
        Self {
            shape,
            t: stacked
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        assert!(dat.is_empty(), "The given data is empty");

        let mut shape = dat[0].shape.clone();
        let n_features: usize = dat[0].t.shape()[1..].iter().product();

        let dim = resolve_negative_index(shape.len(), dim);
        assert!(check_batch_dim_except(&dat, dim), "The batch dimension does not matches");

        // fold
        let out = n_batch(&shape[..dim]);
        let res = n_batch(&shape[dim + 1..]) * n_features;
        let mut k_final = 0;

        let t_vec = dat.into_iter().map(
            |d| {
                let k = d.shape[dim];
                k_final += k;
                d.t.reshape([out, k, res])
            }
        ).collect();

        shape[dim] = k_final;
        let concatanated = Tensor::cat(t_vec, 1).reshape::<D, Shape>(shape.clone().into());
        
        Self {
            shape,
            t: concatanated
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);

        let n_features: usize = self.t.shape()[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = range.iter().len();
        shape[dim] = k_final;

        let t = self.t
            .reshape([out, k , res])
            .narrow(1, range.start, k_final)
            .reshape::<D, Shape>(shape.clone().into());

        Self {
            shape,
            t
        }
    }

    fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);

        let n_features: usize = self.t.shape()[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = idx.shape().len();
        shape[dim] = k_final;

        let t = self.t
            .reshape([out, k , res])
            .select(1, idx)
            .reshape::<D, Shape>(shape.clone().into());

        Self {
            shape,
            t
        }
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let mut shape = data.t.shape();
        shape[0] = shape[0] * capacity;

        let t = Tensor::zeros(shape, &data.t.device());

        let mut shape = data.shape.clone();
        shape.insert(0, capacity);

        Self {
            shape,
            t
        }
    }
}