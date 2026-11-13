//! Test for struct space
use bake_rl::core::env::space::struct_space;
use bake_rl::core::env::space::batched_struct_space;
use crate::core::env::space::BatchedContinuousSpace;
use crate::core::env::space::BatchedDiscreteSpace;
use crate::env::space_transform::one_hot::BatchedOneHotTransform;
use crate::env::space_transform::one_hot::OneHotTransform;

use super::{ContinuousSpace, DiscreteSpace};

///
#[allow(unused)]
#[struct_space]
pub struct MySpace {
    d: DiscreteSpace,
    c: ContinuousSpace<2>,
}

#[allow(unused)]
#[batched_struct_space]
pub struct MySSpace {
    d: BatchedDiscreteSpace,
    c: BatchedContinuousSpace<2>,
}

#[allow(unused)]
fn n() {
    let transformer = MySpace::transform_d(OneHotTransform::new());
    let transformer = BatchedMySpace::transform_d(BatchedOneHotTransform::new());
}

#[allow(unused)]
fn k() {
    let transformer = MySSpace::transform_d(BatchedOneHotTransform::new());
}