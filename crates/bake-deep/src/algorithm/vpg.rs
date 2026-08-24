use std::collections::HashMap;

use burn::{Tensor, config::Config, optim::{GradientsParams, ModuleOptimizer}, tensor::TensorData};

use crate::{approximator::Policy, distribution::Distribution, types::{Batch, Batchable, Recordable}};

#[derive(Debug, Config)]
pub struct Vpg {
    pub gamma: f32,
    pub baseline: Baseline
}

#[derive(Debug, Clone)]
pub struct VpgLoss {
    pub loss: Tensor<1>,
    pub entropy: Tensor<1>,
}

impl Vpg {
    pub fn loss<P: Policy>(config: &Vpg, policy: &P, batch: Batch<P::Obs, <P::Dist as Distribution>::Action, P::Constraint>) -> VpgLoss {
        let len = batch.batch_size();
        let device = batch.device();
        let dist = policy.forward(batch.obss, batch.constraints.into());
        let rewards = batch.rewards.into_data().into_vec().unwrap();
        let mut returns = vec![0f32; len];
        returns[len - 1] = rewards[len - 1];
        for i in (0..(len - 1)).rev() {
            returns[i] = rewards[i] + config.gamma * returns[i + 1]
        }
        
        let returns = Tensor::from_data(TensorData::new(returns, [len]), &device);
        let returns = config.baseline.advantage(returns);

        let log_probs = dist.log_probs(batch.actions);
        let entropy = dist.entropy().mean();
        let loss = -(returns * log_probs).mean();

        VpgLoss { loss, entropy }
    }

    pub fn update<P: Policy>(policy: P, loss: VpgLoss, c_e: f32, lr: f64, opt: &mut ModuleOptimizer) -> P {
        let grads = (loss.loss - c_e * loss.entropy).backward();
        let grads = GradientsParams::from_grads(grads, &policy);
        opt.step(lr, policy, grads)
    }
}

impl Recordable for VpgLoss {
    fn to_record(&self) -> HashMap<&'static str, Tensor<1>> {
        let mut record = HashMap::new();
        record.insert("loss", self.loss.clone().detach());
        record.insert("entropy", self.entropy.clone().detach());
        record
    }
}

/// baseline enum for examining the effects of it
#[derive(Debug, Config)]
pub enum Baseline {
    /// No baseline is applied
    None,
    /// Normalize the return
    Normalized,
    /// Set the mean of returns into zero
    Mean,
}

impl Baseline {
    /// calculate the return with baseline applied
    pub fn advantage(&self, returns: Tensor<1>) -> Tensor<1> {
        match self {
            Baseline::None => returns,
            Baseline::Normalized => {
                (returns.clone() - returns.clone().mean()) / (returns.var(0).sqrt() + 1e-8)
            }
            Baseline::Mean => {
                returns.clone() - returns.clone().mean()
            }
        }
    }
}