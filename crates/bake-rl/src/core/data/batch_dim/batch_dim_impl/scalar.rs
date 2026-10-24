use crate::core::{data::{BatchArray, batch_dim::{convert::AttachBatchDim, ops::n_batch}}};
use ndarray::prelude::*;

impl AttachBatchDim for f32 {
    type Attached = BatchArray<Self>;

    fn attach_batch_dim(self, batch_shape: crate::core::data::BatchShape) -> Self::Attached {
        assert!(batch_shape.is_empty() || n_batch(&batch_shape) == 1, "batch dimension must be all 1 or empty");
        let array = ArrayD::from_elem(batch_shape.clone(), self);
        ArrayD::attach_batch_dim(array, batch_shape)
    }
}

impl AttachBatchDim for i64 {
    type Attached = BatchArray<Self>;

    fn attach_batch_dim(self, batch_shape: crate::core::data::BatchShape) -> Self::Attached {
        assert!(batch_shape.is_empty() || n_batch(&batch_shape) == 1, "batch dimension must be all 1 or empty");
        let array = ArrayD::from_elem(batch_shape.clone(), self);
        ArrayD::attach_batch_dim(array, batch_shape)
    }
}

impl AttachBatchDim for bool {
    type Attached = BatchArray<Self>;

    fn attach_batch_dim(self, batch_shape: crate::core::data::BatchShape) -> Self::Attached {
        assert!(batch_shape.is_empty() || n_batch(&batch_shape) == 1, "batch dimension must be all 1 or empty");
        let array = ArrayD::from_elem(batch_shape.clone(), self);
        ArrayD::attach_batch_dim(array, batch_shape)
    }
}