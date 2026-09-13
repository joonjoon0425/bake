//! LunarLander-v3
//! 

use crate::{env::DiscreteGymnasiumEnvironment, info::DiscreteGymEnvInfo};
/// LunarLander-v3
#[derive(Debug, Clone, Copy)]
pub struct LunarLanderInfo;
impl DiscreteGymEnvInfo<2> for LunarLanderInfo {
    fn name() -> &'static str { "LunarLander-v3" }
    fn obs_shape() -> [usize; 2] { [1, 8] }
    fn n_actions() -> usize { 4 }
}

/// LunarLander-v3
pub type GymLunarLander = DiscreteGymnasiumEnvironment<2, LunarLanderInfo>;