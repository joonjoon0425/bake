//! The data representation marker type and traits
use burn::prelude::*;
use ndarray::{Array, Dim};

use crate::core::data::batch_dim::convert::AttachBatchDim;

/// the marker trait
/// - for unified representation
pub trait Repr: std::fmt::Debug + Copy {
    /// float array type, with fixed rank
    type FloatArray<const D: usize> : AttachBatchDim + Clone + 'static;
    /// integer array type, with fixed rank
    type IntArray<const D: usize> : AttachBatchDim + Clone + 'static;
    /// boolean array type, with fixed rank
    type BoolArray<const D: usize> : AttachBatchDim + Clone + 'static;

    /// float type
    type Float : AttachBatchDim + Clone + 'static;
    /// integer type
    type Int : AttachBatchDim + Clone + 'static;
    /// boolean type
    type Bool : AttachBatchDim + Clone + 'static;
    /// index (usize) type
    type Index : AttachBatchDim + Clone + 'static;

    /// device type
    type Device : Clone + 'static;
}

/// marker for deep rl
#[derive(Debug, Clone, Copy)]
pub struct Vector;
impl Repr for Vector {
    type FloatArray<const D: usize> = Tensor<D>;
    type IntArray<const D: usize> = Tensor<D, Int>;
    type BoolArray<const D: usize> = Tensor<D, Bool>;

    type Float = Tensor<1>;
    type Int = Tensor<1, Int>;
    type Bool = Tensor<1, Bool>;

    type Index = Tensor<1, Int>;

    type Device = Device;
}

/// marker for tabular, linear rl, without batch
#[derive(Debug, Clone, Copy)]
pub struct Single;
impl Repr for Single {
    type FloatArray<const D: usize> = Array<f32, Dim<[usize; D]>>;
    type IntArray<const D: usize> = Array<i64, Dim<[usize; D]>>;
    type BoolArray<const D: usize> = Array<bool, Dim<[usize; D]>>;

    type Float = f32;
    type Int = i64;
    type Bool = bool;

    type Index = usize;

    type Device = ();
}