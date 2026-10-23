use bake_macros::WithBatchDim;
use crate::core::data::batch_dim::{BatchTensor, WithBatchDim};

#[derive(WithBatchDim)]
struct TestStruct {
    f1: BatchTensor<2>,
}