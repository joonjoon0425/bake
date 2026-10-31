//! implementation of empty leaf

use crate::core::data::{BatchOps, batch_ops::helper::*};

impl BatchOps for () {
    fn assign_inplace_batch(&mut self, _bs: &[usize], _dim: isize, _idx: usize, _src: Self, _src_bs: &[usize]) {}
    fn autodiff(self) -> Self { self }
    fn cat_batch(bs: &[Vec<usize>], dim: isize, _dat: Vec<Self>) -> Self {
        bs_after_cat(bs, dim).unwrap();
    }
    fn check_batch(&self, _: &[usize]) -> bool { true }
    fn select_batch(self, bs: &[usize], dim: isize, idx: burn::prelude::Tensor<1, burn::prelude::Int>) -> Self {
        bs_after_select(bs, dim, idx.dims()[0]).unwrap();
    }
    fn slice_batch(self, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> Self {
        bs_after_slice(bs, dim, &range).unwrap();
    }
    fn unbind_batch(self, bs: &[usize], dim: isize) -> Vec<Self> {
        let (_, dim) = bs_after_unbind(bs, dim).unwrap();
        let n = bs[dim];
        vec![(); n]
    }
    fn zeros_like_batch(bs: &[usize], dim: isize, capacity: usize, _src: &Self) -> Self {
        bs_after_zeros_like(bs, dim, capacity).unwrap();
    }
}