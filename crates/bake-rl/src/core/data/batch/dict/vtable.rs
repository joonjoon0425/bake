use crate::core::data::{BatchOps, batch::dict::ErasedBatchOps};
use super::BatchOpsVTable;

impl BatchOpsVTable {
    /// create a new `BatchOpsVTable` of given type
    pub fn new<T: BatchOps>() -> Self {
        Self {
            type_name: std::any::type_name::<T>(),
            check_batch: check_batch_erased::<T>,
            reshape_batch: reshape_batch_erased::<T>,
            cat_batch: cat_batch_erased::<T>,
            unbind_batch: unbind_batch_erased::<T>,
            slice_batch: slice_batch_erased::<T>,
            select_batch: select_batch_erased::<T>,
            assign_inplace_batch: assign_inplace_batch_erased::<T>,
            zeros_like_batch: zeros_like_batch_erased::<T>,
            autodiff: autodiff_erased::<T>,
            clone: clone_erased::<T>,
        }
    }
}

fn check_batch_erased<T: BatchOps>(e: &ErasedBatchOps, bs: &[usize]) -> bool {
    e.downcast_ref::<T>()
    .expect("Dict type mismatch in check_batch")
    .check_batch(bs)
}

fn reshape_batch_erased<T: BatchOps>(e: ErasedBatchOps, bs: &[usize], new_bs: &[usize]) -> ErasedBatchOps {
    Box::new(e.downcast::<T>().expect("Dict type mismatch in reshape_batch")
    .reshape_batch(bs, new_bs))
}

fn cat_batch_erased<T: BatchOps>(bs: &[Vec<usize>], dim: isize, dat: Vec<ErasedBatchOps>) -> ErasedBatchOps {
    let dat: Vec<T> = dat.into_iter().map(|val| *val.downcast::<T>().expect("Dict type mismatch in cat_batch")).collect(); 
    Box::new(T::cat_batch(bs, dim, dat))
}

fn unbind_batch_erased<T: BatchOps>(e: ErasedBatchOps, bs: &[usize], dim: isize) -> Vec<ErasedBatchOps> {
    let unbinded = e.downcast::<T>()
        .expect("Dict type mistmatch in unbind_batch")
        .unbind_batch(bs, dim);
    let mut vec = Vec::with_capacity(unbinded.len());
    for v in unbinded {
        vec.push(Box::new(v) as ErasedBatchOps)
    }
    vec
}

fn slice_batch_erased<T: BatchOps>(e: ErasedBatchOps, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> ErasedBatchOps {
    Box::new(e.downcast::<T>().expect("Dict type mistmatch in slice_batch")
        .slice_batch(bs, dim, range))
}

fn select_batch_erased<T: BatchOps>(e: ErasedBatchOps, bs: &[usize], dim: isize, idx: burn::Tensor<1, burn::prelude::Int>) -> ErasedBatchOps {
    Box::new(e.downcast::<T>().expect("Dict type mistmatch in select_batch")
        .select_batch(bs, dim, idx))
}

fn assign_inplace_batch_erased<T: BatchOps>(e: &mut ErasedBatchOps, bs: &[usize], dim: isize, idx: usize, src: ErasedBatchOps, src_bs: &[usize]) {
    e.downcast_mut::<T>().expect("(self) Dict type mismatch in assign_inplace_batch")
        .assign_inplace_batch(bs, dim, idx, *src.downcast::<T>().expect("(src) Dict type mismatch in assign_inplace_batch"), src_bs);
}

fn zeros_like_batch_erased<T: BatchOps>(bs: &[usize], dim: isize, capacity: usize, src: &ErasedBatchOps) -> ErasedBatchOps {
    Box::new(
        T::zeros_like_batch(
            bs, dim, capacity,
            src.downcast_ref::<T>()
            .expect("Dict type mismatch in zero_like_batch")
        )
    )
}

fn autodiff_erased<T: BatchOps>(e: ErasedBatchOps) -> ErasedBatchOps {
    Box::new(e.downcast::<T>().expect("Dict type mismatch in autodiff")
        .autodiff())
}

fn clone_erased<T: BatchOps>(e: &ErasedBatchOps) -> ErasedBatchOps {
    Box::new(
        e.downcast_ref::<T>().expect("Dict type mismatch in clone").clone()
    )
}