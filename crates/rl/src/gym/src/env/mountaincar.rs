//! MountainCar-v0
//! 

use crate::{env::DiscreteGymnasiumEnvironment, info::DiscreteGymEnvInfo};
/// MountainCar-v0
#[derive(Debug, Clone, Copy)]
pub struct MountainCarInfo;
impl DiscreteGymEnvInfo<2> for MountainCarInfo {
    fn name() -> &'static str { "MountainCar-v0" }
    fn obs_shape() -> [usize; 2] { [1, 2] }
    fn n_actions() -> usize { 3 }
}

/// MountainCar-v0
pub type GymMountainCar = DiscreteGymnasiumEnvironment<2, MountainCarInfo>;