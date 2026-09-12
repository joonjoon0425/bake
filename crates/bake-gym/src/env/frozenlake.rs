//! FrozenLake-v1
//! 

use crate::{env::TabularGymnasiumEnvironment, info::TabularGymEnvInfo};
/// FrozenLake-v1
#[derive(Debug, Clone, Copy)]
pub struct FrozenLakeInfo;
impl TabularGymEnvInfo for FrozenLakeInfo {
    fn name() -> &'static str { "FrozenLake-v1" }
    fn n_obs() -> usize { 16 }
    fn n_actions() -> usize { 4 }
}

/// FrozenLake-v1
pub type GymFrozenLake = TabularGymnasiumEnvironment<FrozenLakeInfo>;