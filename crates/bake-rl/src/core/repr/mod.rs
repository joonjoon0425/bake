//! The data representation marker type and traits
use burn::{prelude::*, tensor::kind::Basic};
use ndarray::ArrayD;

use crate::core::data::{BatchScalar, batch_dim::convert::AttachBatchDim};

/// the Kind of element: float, int and bool
pub trait ElementKind: Basic + Clone + 'static {
    /// the element kind
    type Kind: Default + Clone + Copy + 'static;
}
impl ElementKind for Float { type Kind = f32; }
impl ElementKind for Int { type Kind = i64; }
impl ElementKind for Bool { type Kind = bool; }

/// the marker trait
/// - for unified representation
pub trait Repr: std::fmt::Debug + Copy {
    /// multi-dimensional array type
    type Array<const D: usize, K: ElementKind> : AttachBatchDim + Clone + 'static;
    /// scalar type
    type Scalar<K: ElementKind> : AttachBatchDim + Clone + 'static;
    /// index (usize) type
    type Index : Clone + 'static;
    /// indices type
    type Indices : Clone + 'static;

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
    type Indices = Tensor<1, Int>;
    type Device = Device;
}

/// marker for tabular, linear rl, without batch
#[derive(Debug, Clone, Copy)]
pub struct Single;
impl Repr for Single {
    type Array<const D: usize, K: ElementKind> = ArrayD<K::Kind>;
    type Scalar<K: ElementKind> = BatchScalar<K>;
    type Index = usize;
    type Indices = Vec<usize>;
    type Device = ();
}