use tch::{Tensor};
use super::BatchOps;
use super::helper::*;

impl BatchOps for Tensor {
    fn assign_inplace_batch(&mut self, bs: &[usize], dim: i64, idx: usize, src: &Self, src_bs: &[usize]) {
        assert!(self.check_batch(bs), "BatchOps::assign_inplace_batch: batch shape mismatch in self");
        assert!(src.check_batch(src_bs), "BatchOps::assign_inplace_batch: batch shape mismatch in src");
        
        let dim = check_assign(bs, src_bs, dim, idx).unwrap();
        let n = src_bs[dim];

        self.narrow(dim as i64, idx as i64, n as i64).copy_(src);
    }

    fn cat_batch(bs: &[Vec<usize>], dim: i64, dat: &[&Self]) -> Self {
        let (_, dim) = bs_after_cat(bs, dim).unwrap();
        Tensor::cat(dat, dim as i64)
    }

    fn check_batch(&self, bs: &[usize]) -> bool {
        let size = self.size();
        if size.len() != bs.len() { false }
        else {
            for (i, u) in size.into_iter().zip(bs.iter()) {
                if i != *u as i64 {
                    return false;
                }
            }
            true
        }
    }

    fn reshape_batch(self, bs: &[usize], new_bs: &[usize]) -> Self {
        assert!(self.check_batch(bs), "BatchOps::reshape_batch: batch shape mismatch in self");
        let new_bs: Vec<i64> = new_bs.iter().map(|u| *u as i64).collect();
        self.reshape(new_bs)
    }
}