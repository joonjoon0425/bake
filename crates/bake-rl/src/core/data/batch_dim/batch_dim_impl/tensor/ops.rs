use std::ops::{Add, Sub};

use crate::core::data::{AttachBatchDim, DetachBatchDim, batch_dim::batch_dim_impl::tensor::KindAutoDiff};
use super::BatchTensor;

impl<const D: usize, K: KindAutoDiff> Add for BatchTensor<D, K>
    where burn::Tensor<D, K>: std::ops::Add<Output = burn::Tensor<D, K>>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let (lhs, lhs_bs) = self.detach_batch_dim();
        let (rhs, rhs_bs) = rhs.detach_batch_dim();

        assert!(lhs_bs == rhs_bs, "BatchTensor::add: the rhs and lhs has different batch shape");
        let result = lhs + rhs;
        result.attach_batch_dim(lhs_bs)
    }
}

impl<const D: usize, K: KindAutoDiff> Sub for BatchTensor<D, K>
    where burn::Tensor<D, K>: std::ops::Sub<Output = burn::Tensor<D, K>>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let (lhs, lhs_bs) = self.detach_batch_dim();
        let (rhs, rhs_bs) = rhs.detach_batch_dim();

        assert!(lhs_bs == rhs_bs, "BatchTensor::sub: the rhs and lhs has different batch shape");
        let result = lhs - rhs;
        result.attach_batch_dim(lhs_bs)
    }
}