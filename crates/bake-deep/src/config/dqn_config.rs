use burn::config::Config;
use crate::{algorithm::Dqn, config::OptimizerConfig};

#[derive(Debug, Config)]
pub struct DqnConfig {
    pub dqn: Dqn,
    pub total_episode: usize,
    pub update_freq: usize,
    pub sync_freq: usize,
    pub batch_size: usize,
    
    pub lr: f64,
    pub opt_config: OptimizerConfig,

    pub seed: u64,
    pub log_interval: usize,
}