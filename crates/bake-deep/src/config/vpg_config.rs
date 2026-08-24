use burn::config::Config;
use crate::{algorithm::Vpg, config::OptimizerConfig};

#[derive(Debug, Config)]
pub struct VpgConfig {
    pub vpg: Vpg,
    pub coeff_entropy: f32,
    pub total_episode: usize,

    pub lr: f64,
    pub opt_config: OptimizerConfig,
    
    pub seed: u64,
    pub log_interval: usize,
}
