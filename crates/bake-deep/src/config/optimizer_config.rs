use burn::{config::Config, optim::{AdaGradConfig, AdamConfig, AdamWConfig, ModuleOptimizer, RmsPropConfig, SgdConfig}};

#[derive(Config, Debug)]
pub enum OptimizerConfig {
    Adam(AdamConfig),
    AdamW(AdamWConfig),
    RmsProp(RmsPropConfig),
    Sgd(SgdConfig),
    AdaGrad(AdaGradConfig),
}

impl OptimizerConfig {
    pub fn init(&self) -> ModuleOptimizer {
        match self {
            Self::Adam(c) => c.init(),
            Self::AdamW(c) => c.init(),
            Self::RmsProp(c) => c.init(),
            Self::Sgd(c) => c.init(),
            Self::AdaGrad(c) => c.init(),
        }
    }
}