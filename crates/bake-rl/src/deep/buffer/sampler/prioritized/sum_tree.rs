//! implementation of sum tree for prioritized sampler
//! - reference: https://github.com/thu-ml/tianshou/blob/master/tianshou/data/utils/segtree.py

use burn::prelude::*;

/// implementation of sum tree with `burn::Tensor`
pub struct SumTree {
    size: u64,
    /// the smallest power of 2 greater than `size`
    bound: u64,
    values: Tensor<1>,
    device: Device,
    depth: usize,
}

impl SumTree {
    /// create new `SumTree`
    pub fn new(size: u64, device: &Device) -> Self {
        let mut bound = 1;
        let mut depth = 1;
        while bound < size { bound *= 2; depth += 1; }
        
        let values = Tensor::zeros([2 * bound], device);

        Self {
            size,
            bound,
            depth,
            values,
            device: device.clone()
        }
    }

    #[allow(unused)]
    pub fn len(&self) -> u64 { self.size }

    pub fn total_sum(&self) -> Tensor<1> { self.values.clone().slice(s![1]) }

    /// update priority (values)
    /// - indices: index of update values (index should be given as if the data structure was a normal list)
    /// - values: values which will be switched with original values corresponfing to the indices
    pub fn update(&mut self, indices: Tensor<1, Int>, values: Tensor<1>) {
        assert!(
            indices.clone().greater_equal_elem(0).all().into_scalar()
            && 
            indices.clone().lower_elem(self.size as u64).all().into_scalar(),
            "SumTree::update: indices must be in [0, size)"
        );
        let mut indices = indices + self.bound;
        self.values.inplace(|v| v.select_assign(0, indices.clone(), values, burn::tensor::IndexingUpdateOp::Assign));
        // when same indices exists, it overwrites, using same values
        for _ in 0..self.depth - 1 {
            indices.inplace(|i| i.div_scalar(2));
            self.values.inplace(
                |v| {
                let left = v.clone().select(0, 2 * indices.clone());
                let right = v.clone().select(0, 2 * indices.clone() + 1);
                v.select_assign(0, indices.clone(), left + right, burn::tensor::IndexingUpdateOp::Assign)
            });
        }
    }

    /// sample with distribution p_i/(sum p_i).
    /// # Returns
    /// - returns (sample, idx)
    pub fn sample_stratified(&self, n_samples: i64) -> (Tensor<1>, Tensor<1, Int>) {
        let interval = Tensor::<1, Int>::arange(0..n_samples, &self.device).float();
        let random = Tensor::random([n_samples], burn::tensor::Distribution::Uniform(0., 1.), &self.device);
        let mut query = (interval + random) * self.total_sum() / n_samples as f32;

        let mut descend = Tensor::<1, Int>::ones([n_samples], &self.device);

        for _ in 0..self.depth - 1 {
            let left_idx = descend.clone().mul_scalar(2).clone();
            let left_node = self.values.clone().select(0, left_idx.clone());
            let mask = left_node.clone().greater(query.clone());
            // naive way
            // let query_cloned = query.clone();
            // descend.inplace(|d| d.mask_where(mask.clone(), left_idx));
            // descend.inplace(|d| d.mask_where(mask.clone().bool_not(), right_idx));
            // query.inplace(|q| q.mask_where(mask.bool_not(), query_cloned - left_node));

            // tianshou
            let right = mask.int();
            descend.inplace(|_| left_idx + right.clone());
            query.inplace(|q| q - left_node * right.float())
        }

        (self.values.clone().select(0, descend.clone()), descend - self.bound)
    }
}