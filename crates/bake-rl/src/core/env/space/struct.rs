//! Test for struct space
use bake_rl_macros::{BatchedStructSpace, IntoBatchedStructSpace, StructSpace};

use crate::core::env::space::{BatchedContinuousSpace, BatchedDiscreteSpace};

use super::{ContinuousSpace, DiscreteSpace};

///
#[allow(unused)]
#[derive(StructSpace, IntoBatchedStructSpace, Clone)]
pub struct MySpace {
    d: DiscreteSpace,
    c: ContinuousSpace<2>,
}

///
#[allow(unused)]
#[derive(BatchedStructSpace, Clone)]
pub struct MyBatchedStructSpace {
    d: BatchedDiscreteSpace,
    c: BatchedContinuousSpace<3>,
}
