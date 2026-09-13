//! CartPole-v1
//! 

use crate::{env::DiscreteGymnasiumEnvironment, info::DiscreteGymEnvInfo};
/// CartPole-v1
#[derive(Debug, Clone, Copy)]
pub struct CartPoleInfo;
impl DiscreteGymEnvInfo<2> for CartPoleInfo {
    fn name() -> &'static str { "CartPole-v1" }
    fn obs_shape() -> [usize; 2] { [1, 4] }
    fn n_actions() -> usize { 2 }
}

/// CartPole-v1
pub type GymCartPole = DiscreteGymnasiumEnvironment<2, CartPoleInfo>;