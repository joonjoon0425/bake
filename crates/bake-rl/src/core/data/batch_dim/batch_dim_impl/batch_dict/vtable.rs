use crate::core::data::{BatchShape, WithBatchDim, batch_dim::batch_dim_impl::batch_dict::{Erased, WithBatchDimOpsVTable}};

impl WithBatchDimOpsVTable {
    pub fn new<T: WithBatchDim>() -> Self {
        Self {
            batch_shape: batch_shape::<T>,
            cat: cat::<T>,
            stack: stack::<T>,
            unstack: unstack::<T>,
            reshape: reshape::<T>,
            squeeze: squeeze::<T>,
            unsqueeze: unsqueeze::<T>,
            slice: slice::<T>,
            select: select::<T>,
            assign_inplace: assign_inplace::<T>,
            zeros_like: zeros_like::<T>,
            clone: clone::<T>
        }
    }
}

fn batch_shape<T: WithBatchDim>(v: &Erased) -> BatchShape {
    v.downcast_ref::<T>().expect("BatchDict type mismatch in batch_shape")
        .batch_shape()
}

fn cat<T: WithBatchDim>(v: Vec<Erased>, dim: isize) -> Erased {
    let v: Vec<T> = v.into_iter().map(|val| *val.downcast::<T>().expect("BatchDict type mismatch in cat")).collect();
    Box::new(T::cat(v, dim))
}

fn stack<T: WithBatchDim>(v: Vec<Erased>) -> Erased {
    let v: Vec<T> = v.into_iter().map(|val| *val.downcast::<T>().expect("BatchDict type mismatch in stack")).collect();
    Box::new(T::stack(v))
}

fn unstack<T: WithBatchDim>(v: Erased) -> Vec<Erased> {
    let unstacked = v.downcast::<T>()
        .expect("BatchDict type mismatch in unstack")
        .unstack();
    let mut vec = Vec::with_capacity(unstacked.len());
    for v in unstacked {
        vec.push(Box::new(v) as Erased)
    }
    vec
}

fn reshape<T: WithBatchDim>(v: Erased, shape: BatchShape) -> Erased {
    Box::new(
        v.downcast::<T>().expect("BatchDict type mismatch in reshape")
        .reshape(shape)
    )
}

fn squeeze<T: WithBatchDim>(v: Erased, dim: isize) -> Erased {
    Box::new(
        v.downcast::<T>().expect("BatchDict type mismatch in squeeze")
        .squeeze(dim)
    )
}

fn unsqueeze<T: WithBatchDim>(v: Erased, dim: isize) -> Erased {
    Box::new(
        v.downcast::<T>().expect("BatchDict type mismatch in unsqueeze")
        .unsqueeze(dim)
    )
}

fn slice<T: WithBatchDim>(v: Erased, dim: isize, range: std::ops::Range<usize>) -> Erased {
    Box::new(
        v.downcast::<T>().expect("BatchDict type mismatch in slice")
        .slice(dim, range)
    )
}

fn select<T: WithBatchDim>(v: Erased, dim: isize, idx: burn::Tensor<1, burn::prelude::Int>) -> Erased {
    Box::new(
        v.downcast::<T>().expect("BatchDict type mismatch in select")
        .select(dim, idx)
    )
}

fn assign_inplace<T: WithBatchDim>(dst: &mut Erased, src: Erased, idx: usize) {
    let dst = dst.downcast_mut::<T>().expect("BatchDict type mismatch in assign_inplace");
    let src = *src.downcast::<T>().expect("BatchDict type mismatch in assign_inplace");
    dst.assign_inplace(src, idx);
}

fn zeros_like<T: WithBatchDim>(capacity: usize, data: &Erased) -> Erased {
    let data = data.downcast_ref::<T>().expect("BatchDict type mismatch in zeros_like");
    Box::new(T::zeros_like(capacity, data))
}

fn clone<T: WithBatchDim>(v: &Erased) -> Erased {
    let v = v.downcast_ref::<T>().expect("BatchDict type mismatch in clone");
    Box::new(v.clone())
}