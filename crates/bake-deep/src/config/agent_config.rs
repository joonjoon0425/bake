//! Configuration for agents
use burn::optim::ModuleOptimizer;

/// enum for branching between encoder-sharing and encoder-separated
pub enum ActorCriticConfig {
    /// An encoder-sharing variant
    Shared{
        /// learning rate
        lr: f64,
        /// scales value loss
        c_v: f32,
        /// optimizer
        opt: ModuleOptimizer
    },
    /// An encoder-separated variant
    Separated{
        /// learning rate for policy net
        lr_p: f64,
        /// optimier for policy net
        opt_p: ModuleOptimizer, 
        /// learning rate for value net
        lr_v: f64,
        /// optimier for value net
        opt_v: ModuleOptimizer}
}

impl ActorCriticConfig {
    /// A configuration for encoder-sharing actor-critic method
    pub fn shared(lr: f64, c_v: f32, opt: ModuleOptimizer) -> Self {
        Self::Shared {
            lr,
            c_v,
            opt,
        }
    }
    /// A configuration for encoder-separated actor-critic method
    pub fn separated(lr_p: f64, opt_p: ModuleOptimizer, lr_v: f64, opt_v: ModuleOptimizer) -> Self {
        Self::Separated {
            lr_p,
            opt_p,
            lr_v,
            opt_v,
        }
    }
}