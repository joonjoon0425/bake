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
        let leaf = self.leaf.reshape_batch(&self.bs, &bs);
        Self {
            bs,
            leaf
        }
    }

    /// unsqueeze batch shape
    pub fn unsqueeze(self, dim: isize) -> Self {
        let (bs, _) = bs_after_unsqueeze(&self.bs, dim).unwrap();
        self.reshape(bs)
    }

    /// squeeze batch shape
    pub fn squeeze(self, dim: isize) -> Self {
        let (bs, _) = bs_after_squeeze(&self.bs, dim).unwrap();
        self.reshape(bs)
    }

    /// assign in-place
    pub fn assign_inplace(&mut self, dim: isize, idx: usize, src: Self) {
        self.leaf.assign_inplace_batch(&self.bs, dim, idx, src.leaf, &src.bs);
    }
    
    /// concatenate along the given batch axis;
    pub fn cat(dim: isize, dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "tried to cat empty data");
        let bs: Vec<Vec<usize>> = dat.iter().map(|d| d.bs.clone()).collect();
        let dat = dat.into_iter().map(|b| b.leaf).collect();
        let ret = T::cat_batch(&bs, dim, dat);
        let (bs, _) = bs_after_cat(&bs, dim).unwrap();
        Self {
            bs,
            leaf: ret
        }
    }
    
    /// stack the elements, creating a new batch axis;
    pub fn stack(dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "tried to cat empty data");
        let dat: Vec<Batch<T>> = dat.into_iter().map(|d| d.unsqueeze(0)).collect();
        let bs: Vec<Vec<usize>> = dat.iter().map(
            |d| {
                d.bs.clone()
            }).collect();
        let dat = dat.into_iter().map(|b| b.leaf).collect();
        let ret = T::cat_batch(&bs, 0, dat);
        let bs = bs_after_stack(&bs).unwrap();
        Self {
            bs,
            leaf: ret
        }
    }

    /// unstack along the given batch axis
    pub fn unbind(self, dim: isize) -> Vec<Self> {
        let bs = self.bs;
        let dat = self.leaf;
        let unbinded = dat.unbind_batch(&bs, dim);
        let (bs, _) = bs_after_unbind(&bs, dim).unwrap();
        unbinded.into_iter().map(|u| Self {
            bs: bs.clone(),
            leaf: u
        }).collect()
    }

    /// slice along given axis, with given range
    pub fn slice(self, dim: isize, range: std::ops::Range<usize>) -> Self {
        let bs = self.bs;
        let dat = self.leaf;
        let (bs_after, _) = bs_after_slice(&bs, dim, &range).unwrap();
        let sliced = dat.slice_batch(&bs, dim, range);
        Self {
            bs: bs_after,
            leaf: sliced
        }
    }

    /// gather the elements of batches of given idmension, with given indices;
    pub fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let bs = self.bs;
        let dat = self.leaf;
        let (bs_after, _) = bs_after_select(&bs, dim, idx.dims()[0]).unwrap();
        let selected = dat.select_batch(&bs, dim, idx);
        Self {
            bs: bs_after,
            leaf: selected
        }
    }
    
    /// make zeros with given capacity, along given batch axis
    pub fn zeros_like(dim: isize, capacity: usize, data: &Self) -> Self {
        let bs = data.bs.clone();
        let data_leaf = &data.leaf;
        let z = T::zeros_like_batch(&bs, dim, capacity, data_leaf);
        let (bs, _) = bs_after_zeros_like(&bs, dim, capacity).unwrap();
        Self {
            bs,
            leaf: z
        }
    }
}

/// of course, Batch is also Batchable...
impl<T: BatchOps> BatchOps for Batch<T> {
    fn assign_inplace_batch(&mut self, bs: &[usize], dim: isize, idx: usize, src: Self, src_bs: &[usize]) {
        assert!(bs == self.bs && src_bs == src.bs, "given batch shapes do not equal the batch shapes of self/src");
        self.leaf.assign_inplace_batch(bs, dim, idx, src.leaf, src_bs);
    }

    fn autodiff(self) -> Self {
        Self {
            bs: self.bs,
            leaf: self.leaf.autodiff(),
        }
    }

    fn cat_batch(bs: &[Vec<usize>], dim: isize, dat: Vec<Self>) -> Self {
        let check = bs.into_iter().zip(dat.iter()).all(|(b, d)| {
            if b != d.bs() {
                false
            } else {
                true
            }
        });
        assert!(check, "given batch shape does not equals the batch shape of data");
        let ret = Self::cat(dim, dat);
        ret
    }

    fn check_batch(&self, bs: &[usize]) -> bool {
        self.bs == bs
    }

    fn reshape_batch(self, bs: &[usize], new_bs: &[usize]) -> Self {
        assert!(bs == self.bs, "given batch shape does not equal the batch shape of self");
        self.reshape(new_bs.to_vec())
    }

    fn select_batch(self, bs: &[usize], dim: isize, idx: Tensor<1, Int>) -> Self {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.select(dim, idx);
        ret
    }

    fn slice_batch(self, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> Self {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.slice(dim, range);
        ret
    }

    fn unbind_batch(self, bs: &[usize], dim: isize) -> Vec<Self> {
        assert!(bs == self.bs, "given batch shape does not equals the batch shape of self");
        let ret  = self.unbind(dim);
        ret
    }

    fn zeros_like_batch(bs: &[usize], dim: isize, capacity: usize, src: &Self) -> Self {
        assert!(bs == src.bs(), "given batch shape does not equals the batch shape of src");
        let ret = Self::zeros_like(dim, capacity, src);
        ret
    }
}

pub mod transition;
pub mod dict;

#[cfg(test)]
mod tests {
    //! unit tests for the batch container
    //!
    //! `seq()` is a `[6, 4]` float tensor; with batch shape `[2, 3]`, batch element `(i, j)` is row `i * 3 + j`
    //! and row `r` starts with `4r`.

    use burn::prelude::*;
    use super::Batch;
    use crate::core::data::batch_ops::BatchOps;

    fn dev() -> Device { Device::default() }

    fn seq() -> Tensor<2> {
        Tensor::<1, Int>::arange(0..24, &dev()).float().reshape([6, 4])
    }

    fn firsts(t: Tensor<2>) -> Vec<f32> {
        let n = t.dims()[0];
        t.slice([0..n, 0..1]).try_into_vec_as::<f32>().unwrap()
    }

    fn leaf_firsts(b: Batch<Tensor<2>>) -> Vec<f32> {
        firsts(b.destruct().0)
    }

    fn b23() -> Batch<Tensor<2>> { Batch::new(seq(), vec![2, 3]) }

    // ---------------- construction ----------------

    #[test]
    fn new_and_destruct_roundtrip() {
        let (leaf, bs) = b23().destruct();
        assert_eq!(bs, vec![2, 3]);
        assert_eq!(leaf.dims(), [6, 4]);
    }

    #[test]
    #[should_panic]
    fn new_with_wrong_batch_shape_panics() {
        let _ = Batch::new(seq(), vec![2, 2]);
    }

    #[test]
    fn new_with_empty_batch_shape() {
        let b = Batch::new(Tensor::<2>::zeros([1, 4], &dev()), vec![]);
        assert!(b.bs().is_empty());
    }

    // ---------------- shape-only ops ----------------

    #[test]
    fn reshape_valid() {
        let b = b23().reshape(vec![3, 2]);
        assert_eq!(b.bs(), &[3, 2]);
        assert_eq!(leaf_firsts(b), firsts(seq()));   // data untouched
    }

    #[test]
    #[should_panic]
    fn reshape_invalid_panics() {
        let _ = b23().reshape(vec![4, 2]);
    }

    #[test]
    fn squeeze_unsqueeze() {
        let b = b23().unsqueeze(1);
        assert_eq!(b.bs(), &[2, 1, 3]);
        let b = b.squeeze(1);
        assert_eq!(b.bs(), &[2, 3]);
        let b = b.unsqueeze(-1);
        assert_eq!(b.bs(), &[2, 3, 1]);
    }

    #[test]
    #[should_panic]
    fn squeeze_non_one_panics() {
        let _ = b23().squeeze(0);
    }

    // ---------------- data ops ----------------

    #[test]
    fn slice_updates_shape() {
        let b = b23().slice(1, 0..2);
        assert_eq!(b.bs(), &[2, 2]);
        assert_eq!(leaf_firsts(b), vec![0.0, 4.0, 12.0, 16.0]);
    }

    #[test]
    fn select_updates_shape() {
        let b = b23().select(1, Tensor::from_ints([2, 0], &dev()));
        assert_eq!(b.bs(), &[2, 2]);
        assert_eq!(leaf_firsts(b), vec![8.0, 0.0, 20.0, 12.0]);
    }

    #[test]
    fn cat_different_lengths() {
        let a = b23();
        let c = Batch::new(seq().slice([0..3, 0..4]) + 100.0, vec![1, 3]);
        let b = Batch::cat(0, vec![a, c]);
        assert_eq!(b.bs(), &[3, 3]);
        assert_eq!(leaf_firsts(b), vec![0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 100.0, 104.0, 108.0]);
    }

    #[test]
    #[should_panic]
    fn cat_empty_panics() {
        let _ = Batch::<Tensor<2>>::cat(0, vec![]);
    }

    #[test]
    fn unbind_removes_axis() {
        let parts = b23().unbind(0);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].bs(), &[3]);
        assert_eq!(leaf_firsts(parts[1].clone()), vec![12.0, 16.0, 20.0]);
    }

    #[test]
    fn unbind_to_single_env_convention() {
        // [n_envs] -> n_envs x []
        let parts = Batch::new(seq(), vec![6]).unbind(0);
        assert_eq!(parts.len(), 6);
        assert!(parts.iter().all(|p| p.bs().is_empty()));
    }

    #[test]
    fn stack_unbind_roundtrip() {
        let parts = b23().unbind(0);
        let b = Batch::stack(parts);
        assert_eq!(b.bs(), &[2, 3]);
        assert_eq!(leaf_firsts(b), firsts(seq()));
    }

    #[test]
    fn stack_single_envs() {
        // n_envs x [] -> [n_envs]  (what SyncEnv does)
        let envs: Vec<_> = (0..4)
            .map(|i| Batch::new(Tensor::<2>::zeros([1, 4], &dev()) + i as f32, vec![]))
            .collect();
        let b = Batch::stack(envs);
        assert_eq!(b.bs(), &[4]);
        assert_eq!(leaf_firsts(b), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn zeros_like_capacity() {
        let z = Batch::zeros_like(0, 5, &b23());
        assert_eq!(z.bs(), &[5, 3]);
        assert_eq!(z.destruct().0.dims(), [15, 4]);
    }

    // ---------------- assign_inplace ----------------

    #[test]
    fn assign_inplace_first_axis_index() {
        // replay-buffer style: capacity 4 along the first axis, each slot is [3]
        let mut buf = Batch::zeros_like(0, 4, &b23());              // bs [4, 3]
        buf.assign_inplace(0, 2, b23().slice(0, 1..2));             // write (1, *) at slot 2
        assert_eq!(buf.bs(), &[4, 3]);
        let f = leaf_firsts(buf);
        // slot 2 = rows 6..9
        assert_eq!(&f[6..9], &[12.0, 16.0, 20.0]);
        assert!(f[..6].iter().chain(&f[9..]).all(|v| *v == 0.0));
    }

    #[test]
    fn assign_inplace_inner_axis() {
        let mut buf = Batch::zeros_like(1, 5, &b23());              // bs [2, 5]
        buf.assign_inplace(1, 1, b23());
        assert_eq!(leaf_firsts(buf), vec![0.0, 0.0, 4.0, 8.0, 0.0, 0.0, 12.0, 16.0, 20.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn assign_inplace_overflow_panics() {
        let mut buf = Batch::zeros_like(0, 2, &b23());              // bs [2, 3]
        buf.assign_inplace(0, 1, b23());                            // 2 slots from index 1
    }

    // ---------------- nesting (Batch as a leaf) ----------------

    #[test]
    fn nested_new_requires_equal_shape() {
        let outer = Batch::new(b23(), vec![2, 3]);
        assert_eq!(outer.bs(), &[2, 3]);
    }

    #[test]
    #[should_panic]
    fn nested_shape_mismatch_panics() {
        let _ = Batch::new(b23(), vec![6]);   // inner [2, 3] vs outer [6]
    }

    #[test]
    fn nested_slice_updates_inner_shape() {
        let s = Batch::new(b23(), vec![2, 3]).slice(1, 1..3);
        assert_eq!(s.bs(), &[2, 2]);
        let (inner, _) = s.destruct();
        assert_eq!(inner.bs(), &[2, 2]);
        assert_eq!(leaf_firsts(inner), vec![4.0, 8.0, 16.0, 20.0]);
    }

    #[test]
    fn nested_unbind_and_stack() {
        let parts = Batch::new(b23(), vec![2, 3]).unbind(0);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].bs(), &[3]);
        let b = Batch::stack(parts);
        let (inner, bs) = b.destruct();
        assert_eq!(bs, vec![2, 3]);
        assert_eq!(inner.bs(), &[2, 3]);
    }

    #[test]
    fn nested_check_batch_compares_shape() {
        let inner = b23();
        assert!(inner.check_batch(&[2, 3]));
        assert!(!inner.check_batch(&[6]));
    }

    #[test]
    #[should_panic]
    fn nested_leaf_op_with_wrong_shape_panics() {
        let _ = b23().slice_batch(&[6], 0, 0..1);
    }
}