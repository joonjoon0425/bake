use burn::config::Config;

use crate::{algorithm::A2C, config::OptimizerConfig};

#[derive(Debug, Config)]
pub struct A2CConfig {
    pub a2c: A2C,
    pub coeff_entropy: f32,
    pub total_episode: usize,
    pub rollout_size: usize,

    pub encoder_config: ActorCriticEncoderConfig,

    pub seed: u64,
    pub log_interval: usize,
}

#[derive(Debug, Config)]
pub enum ActorCriticEncoderConfig {
    /// An encoder-sharing variant
    Shared{
        /// learning rate
        lr: f64,
        /// scales value loss
        coeff_critic: f32,
        /// optimizer
        opt_config: OptimizerConfig,
    },
    /// An encoder-separated variant
    Separated{
        /// learning rate for policy net
        lr_actor: f64,
        /// optimizer configuration for policy net
        opt_actor_config: OptimizerConfig, 
        /// learning rate for value net
        lr_critic: f64,
        /// optimizer configuration for value net
        opt_critic_config: OptimizerConfig
    }
}