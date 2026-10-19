//! # `core`: core components of `bake-rl`
//! 

/// the marker trait
/// - for unified representation
pub trait Marker {
    type FloatArray<const D: usize> : Clone + 'static;
    type IntArray<const D: usize> : Clone + 'static;
    type BoolArray<const D: usize> : Clone + 'static;

    type Float : Clone + 'static;
    type Int : Clone + 'static;
    type Bool : Clone + 'static;
}

pub struct Burn;
impl Marker for Burn {
    type FloatArray<const D: usize> = Tensor<D>;
    type IntArray<const D: usize> = Tensor<D, Int>;
    type BoolArray<const D: usize> = Tensor<D, Bool>;

    type Float = Tensor<1>;
    type Int = Tensor<1, Int>;
    type Bool = Tensor<1, Bool>;
}

pub struct NdArray;
impl Marker for NdArray {
    type FloatArray<const D: usize> = Array<f32, Dim<[usize; D]>>;
    type IntArray<const D: usize> = Array<i64, Dim<[usize; D]>>;
    type BoolArray<const D: usize> = Array<bool, Dim<[usize; D]>>;

    type Float = f32;
    type Int = i32;
    type Bool = bool;
}

pub mod buffer;
pub mod constraint;
pub mod contract;
pub mod data;
pub mod distribution;
pub mod env;
pub mod exploration;

use burn::prelude::*;
use ndarray::{Array, Dim};