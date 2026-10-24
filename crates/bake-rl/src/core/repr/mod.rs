//! The data representation marker type and traits
use burn::{prelude::*, tensor::kind::Basic};
use ndarray::{Array, ArrayD, Dim};

use crate::core::data::batch_dim::convert::AttachBatchDim;

pub trait ElementKind: Basic { type Kind; }
impl ElementKind for Float { type Kind = f32; }
impl ElementKind for Int { type Kind = i64; }
impl ElementKind for Bool { type Kind = bool; }

/// the marker trait
/// - for unified representation
pub trait Repr: std::fmt::Debug + Copy {
    /// multi-dimensional array type
    type Array<const D: usize, K: ElementKind>;
    /// scalar type
    type Scalar<K: ElementKind>;
    /// index (usize) type
    type Index : AttachBatchDim + Clone + 'static;

    /// device type
    type Device : Clone + 'static;
}

/// marker for deep rl
#[derive(Debug, Clone, Copy)]
pub struct Vector;
impl Repr for Vector {
    type Array<const D: usize, K: ElementKind> = Tensor<D, K>;
    type Scalar<K: ElementKind> = Tensor<1, K>;
    type Index = Tensor<1, Int>;
    type Device = Device;
}

/// marker for tabular, linear rl, without batch
#[derive(Debug, Clone, Copy)]
pub struct Single;
impl Repr for Single {
    type Array<const D: usize, K: ElementKind> = ArrayD<K::Kind>;
    type Scalar<K: ElementKind> = K::Kind;
    type Index = usize;
    type Device = ();
}