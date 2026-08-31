use burn::config::Config;
use crate::{algorithm::Ppo, config::ActorCriticEncoderConfig};

#[derive(Config, Debug)]
pub struct PpoConfig {
    pub ppo: Ppo,
    pub coeff_entropy: f32,
    pub total_steps: usize,
    pub rollout_size: usize,
    pub minibatch_size: usize,
    pub epoch: usize,

    pub encoder_config: ActorCriticEncoderConfig,

    pub seed: u64,
    pub log_interval: usize,
}