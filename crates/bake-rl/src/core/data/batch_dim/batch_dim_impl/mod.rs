use burn::{prelude::*, tensor::kind::Basic};

use crate::core::data::batch_dim::{BatchShape, WithBatchDim, n_batch, resolve_negative_index};

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
        let dim = resolve_negative_index(self.shape.len(), dim);

        self.shape.insert(dim, 1);

        self
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        let len = n_batch(&self.shape);
        self.t.inplace(|a| a.slice_assign(idx..idx + len, data.t));
    }
}