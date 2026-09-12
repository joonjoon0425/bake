//! CliffWalking-v1
//! 

use crate::{env::TabularGymnasiumEnvironment, info::TabularGymEnvInfo};
/// CliffWalking-v1
#[derive(Debug, Clone, Copy)]
pub struct CliffWalkingInfo;
impl TabularGymEnvInfo for CliffWalkingInfo {
    fn name() -> &'static str { "CliffWalking-v1" }
    fn n_obs() -> usize { 48 }
    fn n_actions() -> usize { 4 }
}

/// CliffWalking-v1
pub type GymCliffWalking = TabularGymnasiumEnvironment<CliffWalkingInfo>;