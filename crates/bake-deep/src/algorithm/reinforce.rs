//! A REINFORCE algorithm implementation
//! 

use burn::{optim::{GradientsParams, ModuleOptimizer}, prelude::*};

use crate::{contract::Policy, data::{Batch, Batchable}, distribution::{Distribution, PossibleConstraint}, logger::ToLog};

/// A state of REINFORCE algorithm
#[derive(Debug, Clone)]
pub struct Reinforce {
    /// discount rate
    pub gamma: f32,
    /// entropy bonus rate
    pub c_e: f32,
    /// baseline for computing the advantage
    pub baseline: Baseline,
}

/// loss struct for REINFORCE algorithm
#[derive(Debug, Clone)]
pub struct ReinforceLoss {
    /// the surrogate loss of REINFORCE algorithm
    pub surrogate_loss: Tensor<1>,
    /// the entropy of policy
    pub entropy: Tensor<1>,
}

impl Reinforce {
    /// compute the loss of REINFORCE algorithm
    pub fn loss<P: Policy>(state: &Reinforce, policy: &P, rollout: Batch<P::Obs, <P::Dist as Distribution>::Sample, impl PossibleConstraint<P::Dist>>) -> ReinforceLoss {
        let len = rollout.len().unwrap();
        let device = rollout.device();
        let dist = policy.forward(rollout.obss, rollout.constraints);
        let mut returns = Tensor::zeros([len], &device);
        returns.assign_inplace(rollout.rewards.clone().slice(len - 1..len), len - 1);
        for i in (0..(len - 1)).rev() {
            let r = rollout.rewards.clone().slice(i..i+1) + state.gamma * returns.clone().slice(i+1..i+2);
            returns.assign_inplace(r, i);
        }

        let returns = state.baseline.advantage(returns).detach();
        let log_probs = dist.log_probs(rollout.actions);
        let entropy = dist.entropy().mean();
        let surrogate_loss = -(returns * log_probs).mean();

        ReinforceLoss { surrogate_loss, entropy }
    }

    /// update the policy with given learning rate and optimizer, with entropy bonus
    pub fn update<P: Policy>(policy: P, loss: ReinforceLoss, c_e: f32, lr: f64, opt: &mut ModuleOptimizer) -> P {
        let grads = (loss.surrogate_loss - c_e * loss.entropy).backward();
        let grads = GradientsParams::from_grads(grads, &policy);
        opt.step(lr, policy, grads)
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "surrogate_loss",
            "entropy"
        ]
    }
}

impl ToLog for ReinforceLoss {
    fn to_log(&self) -> std::collections::HashMap<&'static str, f32> {
        let mut log = std::collections::HashMap::new();
        log.insert("surrogate_loss", self.surrogate_loss.clone().into_scalar());
        log.insert("entropy", self.entropy.clone().into_scalar());

        log
    }
}

/// baseline enum for REINFORCE algorithm
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