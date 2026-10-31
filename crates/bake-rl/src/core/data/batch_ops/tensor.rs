//! BatchOps impl for `burn::Tensor`
use burn::prelude::*;
use burn::tensor::kind::Basic;

use crate::core::data::batch_ops::{
    BatchOps,
    helper::*,
};

trait AutodiffKind: Basic + 'static {
    fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self>;
}
impl AutodiffKind for Float { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t.autodiff() } }
impl AutodiffKind for Int   { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }
impl AutodiffKind for Bool  { fn autodiff_t<const D: usize>(t: Tensor<D, Self>) -> Tensor<D, Self> { t } }

impl<const D: usize, K: AutodiffKind> BatchOps for Tensor<D, K> {
    fn assign_inplace_batch(&mut self, bs: &[usize], dim: isize, idx: usize, src: Self, src_bs: &[usize]) {
        let d = resolve_dim(bs.len(), dim).unwrap();
        assert!(
            check_batch_dim_except(&[bs.to_vec(), src_bs.to_vec()], d),
            "assign_batch: batch shapes {bs:?} and {src_bs:?} differ outside axis {d}"
        );
        assert!(
            idx + src_bs[d] <= bs[d],
            "assign_batch: {} elements from index {idx} exceed axis {d} of size {}", src_bs[d], bs[d]
        );
        let dims = self.dims();
        assert!(dims[1..] == src.dims()[1..], "assign_batch: feature dims {:?} and {:?} differ", &dims[1..], &src.dims()[1..]);
        assert!(self.check_batch(bs) && src.check_batch(src_bs), "assign_batch: tensor rows do not match the given batch shapes");

        // fold: [out, k, res]
        let n_features: usize = dims[1..].iter().product();
        let out: usize = bs[..d].iter().product();
        let res = bs[d + 1..].iter().product::<usize>() * n_features;
        let k = bs[d];
        let k_src = src_bs[d];

        self.inplace(|a| {
            a.reshape([out, k, res])
                .slice_assign([0..out, idx..idx + k_src, 0..res], src.reshape([out, k_src, res]))
                .reshape(dims)
        });
    }

    fn autodiff(self) -> Self {
        K::autodiff_t(self)
    }

    fn cat_batch(bs: &[Vec<usize>], dim: isize, dat: Vec<Self>) -> Self {
        let (shape, dim) = bs_after_cat(&bs, dim).unwrap();
        let ks: Vec<usize> = bs.iter().map(|b| b[dim]).collect();
        let mut dims = dat[0].dims();
        let n_features: usize = dims[1..].iter().product();
        // fold
        let out = shape[..dim].iter().product();
        let res = shape[dim + 1..].iter().product::<usize>() * n_features;
        let t_vec = dat.into_iter().zip(ks).map(
            |(t, k)| {
                t.reshape([out, k, res])
            }
        ).collect();

        // first dimension (batch axis) as # of batches
        dims[0] = shape.iter().product();

        let cat = Tensor::cat(t_vec, 1).reshape(dims);
        cat
    }

    fn check_batch(&self, bs: &[usize]) -> bool {
        self.dims()[0] == bs.iter().product::<usize>()
    }

    fn select_batch(self, bs: &[usize], dim: isize, idx: Tensor<1, Int>) -> Self {
        let (_, dim) = bs_after_select(bs, dim, idx.dims()[0]).unwrap();
        let len = idx.dims()[0];
        
        let mut dims = self.dims();
        let n_features: usize = dims[1..].iter().product();
        
        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res: usize = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let mut shape = bs.to_vec(); shape[dim] = len;
        dims[0] = shape.iter().product();
        let t = 
            self
            .reshape([out, k, res])
            .select(1, idx)
            .reshape(dims);
        t
    }

    fn slice_batch(self, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> Self {
        let (_, dim) = bs_after_slice(bs, dim, &range).unwrap();
        let len = range.end - range.start;

        let mut dims = self.dims();
        let n_features: usize = dims[1..].iter().product();

        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res: usize = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let mut shape = bs.to_vec(); shape[dim] = len;
        dims[0] = shape.iter().product();
        let t =
            self
            .reshape([out, k, res])
            .narrow(1, range.start, len)
            .reshape(dims);
        
        t
    }

    fn unbind_batch(self, bs: &[usize], dim: isize) -> Vec<Self> {
        assert!(!bs.is_empty(), "batch shape is empty");
        let (_, dim) = bs_after_unbind(bs, dim).unwrap();
        
        let mut dims = self.dims();
        let n_features: usize = dims[1..].iter().product();

        let out: usize = bs[..dim].iter().product();
        let k = bs[dim];
        let res = bs[dim + 1..].iter().product::<usize>() * n_features;
        
        let parts =
            self
            .reshape([out, k, res])
            .split(1, 1);

        let mut shape = bs.to_vec(); shape.remove(dim);
        dims[0] = shape.iter().product();

        let parts = parts.into_iter().map(|t| t.reshape(dims)).collect();
        parts
    }

    fn zeros_like_batch(bs: &[usize], dim: isize, capacity: usize, src: &Self) -> Self {
        let (_, dim) = bs_after_zeros_like(bs, dim, capacity).unwrap();
        let mut shape = bs.to_vec(); shape[dim] = capacity;
        let mut tdims = src.dims(); tdims[0] = shape.iter().product();

        let t = Tensor::zeros(tdims, (&src.device(), src.dtype()));

        t
    }
}
