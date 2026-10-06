//! Acrobot-v1
//! 

use crate::{env::DiscreteGymnasiumEnvironment, info::DiscreteGymEnvInfo};
/// Acrobot-v1
#[derive(Debug, Clone, Copy)]
pub struct AcrobotInfo;
impl DiscreteGymEnvInfo<2> for AcrobotInfo {
    fn name() -> &'static str { "Acrobot-v1" }
    fn obs_shape() -> [usize; 2] { [1, 6] }
    fn n_actions() -> usize { 3 }
}

/// Acrobot-v1
pub type GymAcrobot = DiscreteGymnasiumEnvironment<2, AcrobotInfo>;