use burn::{prelude::*, tensor::kind::Basic};

use crate::core::data::batch_dim::{BatchShape, WithBatchDim, check_batch_dim, check_batch_dim_except, n_batch, resolve_negative_index};

/// `burn::Tensor`, with batch dimension
#[derive(Debug, Clone)]
pub struct BatchTensor<const D: usize, K: Basic = Float> {
    shape: BatchShape,
    t: Tensor<D, K>,
}

impl<const D: usize, K: Basic> BatchTensor<D, K> {
    /// create a new `BatchTensor` with empty batch dimension, from given tensor
    pub fn new(tensor: Tensor<D, K>) -> Self {
        debug_assert_eq!(tensor.dims()[0], 1, "empty batch shape requires dim 0 == 1");

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
        assert_eq!(&data.shape[1..], &self.shape[1..], "batch dims except axis 0 must match");
        assert!(idx + data.shape[0] <= self.shape[0], "out of range");
        
        let row = n_batch(&self.shape[1..]);
        let k = data.shape[0];
        self.t.inplace(|a| a.slice_assign((idx * row)..(idx + k) * row, data.t));
    }

    fn stack(dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "Given data vector is empty");
        assert!(check_batch_dim(&dat), "The batch dimension does not matches");

        let n = dat.len();
        let mut shape = dat[0].shape.clone();
        shape.insert(0, n);
        let inner = dat.into_iter().map(|d| d.t).collect();
        let stacked = Tensor::cat(inner, 0);
        Self {
            shape,
            t: stacked
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        assert!(!dat.is_empty(), "The given data is empty");

        let mut shape = dat[0].shape.clone();
        let mut dims = dat[0].t.dims();
        let n_features: usize = dims[1..].iter().product();

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
        dims[0] = n_batch(&shape);
        let concatanated = Tensor::cat(t_vec, 1).reshape(dims);
        
        Self {
            shape,
            t: concatanated
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);
        let mut dims = self.t.dims();
        let n_features: usize = dims[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = range.iter().len();
        shape[dim] = k_final;
        dims[0] = n_batch(&shape);
        let t = self.t
            .reshape([out, k , res])
            .narrow(1, range.start, k_final)
            .reshape(dims);

        Self {
            shape,
            t
        }
    }

    fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);
        let mut dims = self.t.dims();
        let n_features: usize = dims[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = idx.dims()[0];
        shape[dim] = k_final;
        dims[0] = n_batch(&shape);

        let t = self.t
            .reshape([out, k , res])
            .select(1, idx)
            .reshape(dims);

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

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::tensor::kind::Basic;
    use crate::core::data::batch_dim::{BatchShape, BatchTensor, n_batch, WithBatchDim};
    fn d() -> Device { Device::default() }

    fn bt<const D: usize, K: Basic>(shape: BatchShape) -> BatchTensor<D, K> {
        let mut s = [2; D];
        let n = n_batch(&shape);
        s[0] = n;
        let t = Tensor::ones(s, &d());
        BatchTensor {
            shape,
            t
        }
    }

    #[test]
    fn reshape() {
        let bt = bt::<2, Float>(vec![2, 3]);
        let bt = bt.reshape(vec![3, 2]);
        assert!(vec![3, 2] == *bt.batch_shape(), "The reshape did not worked as expected");
        let bt = bt.reshape(vec![1, 6]);
        assert!(vec![1, 6] == *bt.batch_shape(), "The reshape did not worked as expected");
    }

    #[test]
    #[should_panic]
    fn reshape_panic() {
        let bt = bt::<3, Float>(vec![3, 4, 2]);
        bt.reshape(vec![6, 2]);
    }

    #[test]
    fn squeeze() {
        let bt = bt::<4, Float>(vec![2, 3, 4, 1]);
        let bt = bt.squeeze(3);
        assert!(vec![2, 3, 4] == *bt.batch_shape(), "squeeze did not worked as expected");
    }

    #[test]
    fn squeeze_neg_idx() {
        let bt = bt::<3, Float>(vec![2, 1, 2]);
        let bt = bt.squeeze(-2);
        assert!(vec![2, 2] == *bt.batch_shape(), "squeeze did not worked as expected with negative indexing");
    }

    #[test]
    #[should_panic]
    fn squeeze_panic() {
        let bt = bt::<3, Float>(vec![2, 1, 2]);
        bt.squeeze(0);
    }

    #[test]
    fn unsqueeze() {
        let bt = bt::<2, Float>(vec![2, 5]);
        let bt = bt.unsqueeze(0);
        assert!(vec![1, 2, 5] == *bt.batch_shape(), "unsqueeze did not work as expected");
    }

    #[test]
    fn unsqueeze_neg_idx() {
        let bt = bt::<3, Float>(vec![2, 5, 2]);
        let bt = bt.unsqueeze(-2);
        println!("after unsqueeze: {:?}", bt.batch_shape());
        assert!(vec![2, 5, 1, 2] == *bt.batch_shape(), "unsqueeze did not work as expected with negative indexing");
    }

    #[test]
    #[should_panic]
    fn unsqueeze_panic() {
        let bt = bt::<3, Float>(vec![2, 5, 2]);
        bt.unsqueeze(4);
    }

    #[test]
    fn stack() {
        let bts = vec![bt::<2, Float>(vec![2, 3]); 3];
        let stacked = BatchTensor::stack(bts);
        assert!(vec![3, 2, 3] == *stacked.batch_shape(), "stack did not work as expected");
    }

    #[test]
    #[should_panic]
    fn stack_panic() {
        let bts = vec![
            bt::<2, Float>(vec![2, 1]),
            bt::<2, Float>(vec![2, 2])
        ];
        BatchTensor::stack(bts);
    }

    #[test]
    fn cat() {
        let bts = vec![
            bt::<4, Float>(vec![3, 1, 2, 3]),
            bt::<4, Float>(vec![3, 4, 2, 3])
        ];
        let cat = BatchTensor::cat(bts, 1);
        assert!(vec![3, 5, 2, 3] == *cat.batch_shape(), "cat did not work as expected");
    }

    #[test]
    #[should_panic]
    fn cat_panic() {
        let bts = vec![
            bt::<4, Float>(vec![3, 2, 1, 2]),
            bt::<4, Float>(vec![3, 1, 2, 1])
        ];
        BatchTensor::cat(bts, 0);
    }

    
}