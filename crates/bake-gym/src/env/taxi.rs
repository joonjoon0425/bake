//! Taxi-v4
//! 

use crate::{env::TabularGymnasiumEnvironment, info::TabularGymEnvInfo};
/// Taxi-v4
#[derive(Debug, Clone, Copy)]
pub struct TaxiInfo;
impl TabularGymEnvInfo for TaxiInfo {
    fn name() -> &'static str { "Taxi-v4" }
    fn n_obs() -> usize { 500 }
    fn n_actions() -> usize { 6 }
}

/// Taxi-v4
pub type GymTaxi = TabularGymnasiumEnvironment<TaxiInfo>;