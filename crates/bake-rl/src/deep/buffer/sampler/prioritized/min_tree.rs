//! implementation of minimum tree for prioritized sampler

use burn::prelude::*;

/// implementation of minimum tree with `burn::Tensor`
pub struct MinTree {
    size: u64,
    /// the smallest power of 2 greater than `size`
    bound: u64,
    values: Tensor<1>,
    #[allow(unused)]
    device: Device,
    depth: usize,
}

impl MinTree {
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

    pub fn min(&self) -> Tensor<1> { self.values.clone().slice(s![1]) }

    /// update priority (values)
    /// - indices: index of update values (index should be given as if the data structure was a normal list)
    /// - values: values which will be switched with original values corresponfing to the indices
    pub fn update(&mut self, indices: Tensor<1, Int>, values: Tensor<1>) {
        assert!(
            indices.clone().greater_equal_elem(0).all().into_scalar()
            && 
            indices.clone().lower_elem(self.size as u64).all().into_scalar(),
            "MinTree::update: indices must be in [0, size)"
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
                let mask = left.clone().lower(right.clone());
                let min = left.mask_where(mask, right);
                v.select_assign(0, indices.clone(), min, burn::tensor::IndexingUpdateOp::Assign)
            });
        }
    }
}