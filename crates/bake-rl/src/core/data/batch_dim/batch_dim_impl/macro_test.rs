use bake_macros::WithBatchDim;
use crate::core::data::batch_dim::{BatchEmpty, BatchTensor};

#[derive(WithBatchDim, Clone)]
struct _TestStruct {
    f3: BatchEmpty<()>,
    f1: BatchTensor<2>,
    f2: BatchTensor<3>,
}

#[cfg(test)]
mod test {
    use crate::core::data::batch_dim::batch_dim_impl::macro_test::_TestStruct;
    use crate::core::data::batch_dim::{BatchEmpty, BatchTensor, WithBatchDim};
    use burn::prelude::*;

    fn ts() -> _TestStruct {
        let device = Device::default();
        
        let t = _TestStruct {
            f1: BatchTensor::with_empty_batch_dim(Tensor::from_floats([[1., 2., 3.]], &device)),
            f2: BatchTensor::with_empty_batch_dim(Tensor::from_floats(
                [
                    [
                        [1., 2., 3.],
                        [3., 1., 2.],
                        [4., 2., 1.]
                    ],
                ],
                &device)
            ),
            f3: BatchEmpty::with_empty_batch_dim(),
        };

        WithBatchDim::zeros_like(30, &t).reshape(vec![5, 6])
    }

    #[test]
    fn shape() {
        let ts = ts();
        assert_eq!(vec![5, 6], *ts.batch_shape(), "reshape did not work as expected");
    }

    #[test]
    #[should_panic]
    fn reshape_panic() {
        let ts = ts();
        ts.reshape(vec![1, 2]);
    }
}