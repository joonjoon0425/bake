//! Dictionary which can contain any type of `WithBatchDim` types

use std::{any::Any, collections::BTreeMap, range::Range};
use crate::core::data::BatchShape;

/// Dictionary with batch dimension
#[derive(Debug, Clone)]
pub struct BatchDict {
    dict: BTreeMap<&'static str, BatchDictLeaf>
}

pub type Erased = Box<dyn Any + Sync + Send>;
/// the leaf struct of `BatchDict`
#[derive(Debug)]
pub struct BatchDictLeaf {
    value: Erased,
    type_name: &'static str,
    vtable: WithBatchDimOpsVTable,
}

/// handmade vtable for functions of `WithBatchDim`
#[derive(Debug, Clone)]
struct WithBatchDimOpsVTable {
    batch_shape: fn(&Erased) -> BatchShape,
    cat: fn(Vec<Erased>, isize) -> Erased,
    stack: fn(Vec<Erased>) -> Erased,
    unstack: fn(Erased) -> Vec<Erased>,
    reshape: fn(Erased, BatchShape) -> Erased,
    squeeze: fn(Erased, isize) -> Erased,
    unsqueeze: fn(Erased, isize) -> Erased,
    slice: fn(Erased, isize, Range<usize>) -> Erased,
    select: fn(Erased, isize, burn::Tensor<1, burn::prelude::Int>) -> Erased,
    assign_inplace: fn(&mut Erased, Erased, usize),
    zeros_like: fn(usize, &Erased) -> Erased,
    clone: fn(&Erased) -> Erased,
}

mod dict;
mod vtable;