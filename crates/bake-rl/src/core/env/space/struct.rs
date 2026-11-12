//! Test for struct space
use bake_rl_macros::{BatchedStructSpace, IntoBatchedStructSpace, StructSpace};
use crate::core::env::space::{BatchedContinuousSpace, BatchedDiscreteSpace, IntoBatchedSpace};

use super::{ContinuousSpace, DiscreteSpace};

///
#[allow(unused)]
#[derive(StructSpace, IntoBatchedStructSpace, Debug, Clone)]
pub struct MySpace {
    d: DiscreteSpace,
    c: ContinuousSpace<2>,
}

///
#[allow(unused)]
#[derive(BatchedStructSpace, Debug, Clone)]
pub struct MyBatchedStructSpace {
    d: BatchedDiscreteSpace,
    c: BatchedContinuousSpace<3>,
}

///
#[allow(unused)]
#[derive(StructSpace, IntoBatchedStructSpace, Debug, Clone)]
pub struct NestedStructSpace {
    d: MySpace,
    c: ContinuousSpace<2>,
}

///
#[allow(unused)]
#[derive(StructSpace, IntoBatchedStructSpace, Debug, Clone)]
pub struct GenericSpace<S1: IntoBatchedSpace, S2: IntoBatchedSpace> {
    s1: S1,
    s2: S2,
}