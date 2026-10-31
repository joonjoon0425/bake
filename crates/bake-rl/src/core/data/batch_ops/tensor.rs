//! BatchOps impl for `burn::Tensor`
use burn::prelude::*;
use burn::tensor::kind::Basic;

use crate::core::data::batch_ops::{
    BatchOps,
    helper::*,
};

trait AutodiffKind: Basic {
    fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self>;
}
impl AutodiffKind for Float { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t.autodiff() } }
impl AutodiffKind for Int   { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }
impl AutodiffKind for Bool  { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }

impl<const D: usize, K: AutodiffKind> BatchOps for Tensor<D, K> {
    fn assign_inplace(&mut self, src: Self, idx: usize) {
        assert!(self.dims().iter().product::<usize>() >= src.dims().iter().product(), "given `src` is bigger than `self`");
        let k = src.shape()[0];
        self.inplace(|a| a.slice_assign(idx..idx + k, src));
    }

    fn autodiff(self) -> Self {
        K::autodiff_t(self)
    }

    fn cat_batch(bs: Vec<Vec<usize>>, dim: isize, dat: Vec<Self>) -> (Self, Vec<usize>) {
        let (bs, dim) = check_cat_batch(bs, dim, &dat).unwrap();
        let len = dat.len();
        let mut dims = dat[0].dims();
        let n_features: usize = dims[1..].iter().product();
        // fold
        let out = bs[..dim].iter().product();
        let res = bs[dim + 1..].iter().product::<usize>() * n_features;
        let t_vec = dat.into_iter().map(
            |d| {
                let k = bs[dim];
                d.reshape([out, k, res])
            }
        ).collect();

        // concatenated shape
        let mut shape = bs;
        shape[dim] = shape[dim] * len;
        // first dimension (batch axis) as # of batches
        dims[0] = shape.iter().product();

        let cat = Tensor::cat(t_vec, 1).reshape(dims);
        (cat, shape)
    }

    fn select_batch(self, bs: Vec<usize>, dim: isize, idx: Tensor<1, Int>) -> (Self, Vec<usize>) {
        let dim = resolve_negative_index(bs.len(), dim).unwrap();
        let len = idx.dims()[0];
        
        let mut dims = self.dims();
        let n_features: usize = dims[..1].iter().product();
        
        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res: usize = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let mut shape = bs; shape[dim] = len;
        dims[0] = shape.iter().product();
        let t = 
            self
            .reshape([out, k, res])
            .select(1, idx)
            .reshape(dims);

        

        (t, shape)
    }

    fn slice_batch(self, bs: Vec<usize>, dim: isize, range: std::ops::Range<usize>) -> (Self, Vec<usize>) {
        let dim = resolve_negative_index(bs.len(), dim).unwrap();
        let len = range.end - range.start;

        let mut dims = self.dims();
        let n_features: usize = dims[..1].iter().product();

        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res: usize = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let mut shape = bs; shape[dim] = len;
        dims[0] = shape.iter().product();
        let t =
            self
            .reshape([out, k, res])
            .narrow(1, range.start, len)
            .reshape(dims);
        

        (t, shape)
    }

    fn unbind_batch(self, bs: Vec<usize>, dim: isize) -> (Vec<Self>, Vec<usize>) {
        assert!(!bs.is_empty(), "batch shape is empty");
        let dim = resolve_negative_index(bs.len(), dim).unwrap();
        
        let mut dims = self.dims();
        let n_features: usize = dims[1..].iter().product();

        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let parts =
            self
            .reshape([out, k, res])
            .split(1, 1);

        let mut shape = bs; shape[dim] = 1;
        dims[0] = shape.iter().product();

        let parts = parts.into_iter().map(|t| t.reshape(dims)).collect();
        (parts, shape)
    }

    fn zeros_like_batch(bs: Vec<usize>, dim: isize, capacity: usize, src: &Self) -> (Self, Vec<usize>) {
        let dim = resolve_negative_index(bs.len(), dim).unwrap();
        let mut shape = bs; shape[dim] = capacity;
        let mut tdims = src.dims(); tdims[0] = shape.iter().product();

        let t = Tensor::zeros(tdims, (&src.device(), src.dtype()));

        (t, shape)
    }
}