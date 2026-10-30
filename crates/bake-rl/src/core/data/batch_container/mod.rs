//! trait for containers with batch dimension

use burn::prelude::*;
use crate::core::data::batch_ops::{BatchOps, helper::*};

/// trait for container with batch shape information
pub trait BatchContainer: BatchOps {
    /// the batch shape
    fn bs(&self) -> Vec<usize>;
    /// reshape batch shape
    fn reshape(self, bs: Vec<usize>) -> Self;
    /// unsqueeze batch shape
    fn unsqueeze(self, dim: isize) -> Self {
        let (bs, _) = check_unsqueeze(self.bs(), dim).unwrap();
        self.reshape(bs)
    }

    /// squeeze batch shape
    fn squeeze(self, dim: isize) -> Self {
        let (bs, _) = check_squeeze(self.bs(), dim).unwrap();
        self.reshape(bs)
    }
    
    /// concatenate along the given batch axis;
    fn cat(dim: isize, dat: Vec<Self>) -> Self {
        let bs = dat.get(0).expect("tried to catenate empty data").bs();
        let (ret, bs) = Self::cat_batch(bs, dim, dat);
        ret.reshape(bs)
    }
    
    /// stack the elements, creating a new batch axis;
    fn stack(dat: Vec<Self>) -> Self {
        let mut bs = dat.get(0).expect("tried to stack empty data").bs();
        bs.insert(0, 1);
        let (ret, bs) = Self::cat_batch(bs, 0, dat);
        ret.reshape(bs)
    }

    /// unstack along the given batch axis
    fn unbind(self, dim: isize) -> Vec<Self> {
        let bs = self.bs();
        let (unbinded, bs) = self.unbind_batch(bs, dim);
        unbinded.into_iter().map(|u| u.reshape(bs.clone())).collect()
    }

    /// slice along given axis, with given range
    fn slice(self, dim: isize, range: std::ops::Range<usize>) -> Self {
        let bs = self.bs();
        let (sliced, bs) = self.slice_batch(bs, dim, range);
        sliced.reshape(bs)
    }

    /// gather the elements of batches of given idmension, with given indices;
    fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let bs = self.bs();
        let (selected, bs) = self.select_batch(bs, dim, idx);
        selected.reshape(bs)
    }
    
    /// make zeros with given capacity, along given batch axis
    fn zeros_like(dim: isize, capacity: usize, data: &Self) -> Self {
        let (z, bs) = Self::zeros_like_batch(data.bs(), dim, capacity, data);
        z.reshape(bs)
    }
}