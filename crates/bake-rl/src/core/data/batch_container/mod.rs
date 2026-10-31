//! containers with batch dimension

use burn::prelude::*;
use crate::core::data::batch_ops::{BatchOps, helper::*};

/// container with batch shape information
#[derive(Debug, Clone)]
pub struct Batch<T: BatchOps> {
    leaf: T,
    bs: Vec<usize>,
}

impl<T: BatchOps> Batch<T> {
    /// create a new `Batch`, using given leaf and batch shape
    pub fn new(leaf: T, bs: Vec<usize>) -> Self {
        assert!(leaf.check_batch(&bs), "tried to create `Batch` with invalid batch shape");
        Self {
            leaf,
            bs
        }
    }
    /// break down the `Batch`, returning the leaf and batch shape
    pub fn destruct(self) -> (T, Vec<usize>) {
        (self.leaf, self.bs)
    }
    /// returns the batch shape
    pub fn bs(&self) -> &[usize] { &self.bs }
    /// reshape batch shape
    pub fn reshape(self, bs: Vec<usize>) -> Self {
        assert!(check_reshape(&self.bs, &bs), "invalid reshape shape given");
        Self {
            bs,
            ..self
        }
    }

    /// unsqueeze batch shape
    pub fn unsqueeze(self, dim: isize) -> Self {
        let (bs, _) = check_unsqueeze(self.bs, dim).unwrap();
        Self {
            bs,
            ..self
        }
    }

    /// squeeze batch shape
    pub fn squeeze(self, dim: isize) -> Self {
        let (bs, _) = check_squeeze(self.bs, dim).unwrap();
        Self {
            bs,
            ..self
        }
    }
    
    /// concatenate along the given batch axis;
    pub fn cat(dim: isize, dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "tried to cat empty data");
        let bs = dat.iter().map(|d| d.bs.clone()).collect();
        let dat = dat.into_iter().map(|b| b.leaf).collect();
        let (ret, bs) = T::cat_batch(bs, dim, dat);
        Self {
            bs,
            leaf: ret
        }
    }
    
    /// stack the elements, creating a new batch axis;
    pub fn stack(dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "tried to cat empty data");
        let bs = dat.iter().map(
            |d| {
                let mut bs = d.bs.clone();
                bs.insert(0, 1);
                bs
            }).collect();
        let dat = dat.into_iter().map(|b| b.leaf).collect();
        let (ret, bs) = T::cat_batch(bs, 0, dat);
        Self {
            bs,
            leaf: ret
        }
    }

    /// unstack along the given batch axis
    pub fn unbind(self, dim: isize) -> Vec<Self> {
        let bs = self.bs;
        let dat = self.leaf;
        let (unbinded, bs) = dat.unbind_batch(bs, dim);
        unbinded.into_iter().map(|u| Self {
            bs: bs.clone(),
            leaf: u
        }).collect()
    }

    /// slice along given axis, with given range
    pub fn slice(self, dim: isize, range: std::ops::Range<usize>) -> Self {
        let bs = self.bs;
        let dat = self.leaf;
        let (sliced, bs) = dat.slice_batch(bs, dim, range);
        Self {
            bs,
            leaf: sliced
        }
    }

    /// gather the elements of batches of given idmension, with given indices;
    pub fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let bs = self.bs;
        let dat = self.leaf;
        let (selected, bs) = dat.select_batch(bs, dim, idx);
        Self {
            bs,
            leaf: selected
        }
    }
    
    /// make zeros with given capacity, along given batch axis
    pub fn zeros_like(dim: isize, capacity: usize, data: &Self) -> Self {
        let bs = data.bs.clone();
        let data_leaf = &data.leaf;
        let (z, bs) = T::zeros_like_batch(bs, dim, capacity, data_leaf);
        Self {
            bs,
            leaf: z
        }
    }
}

/// of course, Batch is also Batchable...
impl<T: BatchOps> BatchOps for Batch<T> {
    fn assign_inplace(&mut self, src: Self, idx: usize) {
        self.leaf.assign_inplace(src.leaf, idx);
    }

    fn autodiff(self) -> Self {
        Self {
            bs: self.bs,
            leaf: self.leaf.autodiff(),
        }
    }

    fn cat_batch(bs: Vec<Vec<usize>>, dim: isize, dat: Vec<Self>) -> (Self, Vec<usize>) {
        let check = bs.into_iter().zip(dat.iter()).all(|(b, d)| {
            if b != d.bs() {
                false
            } else {
                true
            }
        });
        assert!(check, "given batch shape does not equals the batch shape of data");
        let ret = Self::cat(dim, dat);
        let bs = ret.bs.clone();
        (ret, bs)
    }

    fn check_batch(&self, bs: &[usize]) -> bool {
        self.bs == bs
    }

    fn select_batch(self, bs: Vec<usize>, dim: isize, idx: Tensor<1, Int>) -> (Self, Vec<usize>) {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.select(dim, idx);
        let bs = ret.bs.clone();
        (ret, bs)
    }

    fn slice_batch(self, bs: Vec<usize>, dim: isize, range: std::ops::Range<usize>) -> (Self, Vec<usize>) {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.slice(dim, range);
        let bs = ret.bs.clone();
        (ret, bs)
    }

    fn unbind_batch(self, bs: Vec<usize>, dim: isize) -> (Vec<Self>, Vec<usize>) {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.unbind(dim);
        let bs = ret[0].bs.clone();
        (ret, bs)
    }

    fn zeros_like_batch(bs: Vec<usize>, dim: isize, capacity: usize, src: &Self) -> (Self, Vec<usize>) {
        assert!(bs == src.bs(), "given batch shape does not equals the batch shape of src");
        let ret = Self::zeros_like(dim, capacity, src);
        let bs = ret.bs.clone();
        (ret, bs)
    }
}